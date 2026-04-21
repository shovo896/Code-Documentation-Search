import hashlib
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import GitLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

load_dotenv()


PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME = "code-doc-search-openai"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
MAX_FILE_SIZE_KB = 1500
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

ALLOWED_EXTENSIONS = {".py", ".md", ".rst", ".txt", ".js", ".ts", ".java", ".go"}
SKIP_DIRS = {"node_modules", "__pycache__", ".git", "dist", "build", "venv", ".venv"}


def normalize_repo_url(repo_url: str) -> str:
    """Return a normalized public GitHub repository URL."""
    repo_url = repo_url.strip().removesuffix("/")
    repo_url = repo_url.removesuffix(".git")

    match = re.fullmatch(r"https://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)", repo_url)
    if not match:
        raise ValueError("Please provide a public GitHub repository URL like https://github.com/owner/repo")

    owner, repo = match.groups()
    return f"https://github.com/{owner}/{repo}"


def namespace_for_repo(repo_url: str, branch: str = "main") -> str:
    """Create a stable Pinecone namespace for a repository and branch."""
    normalized = normalize_repo_url(repo_url)
    branch = (branch or "main").strip()
    digest = hashlib.sha1(f"{normalized}:{branch}".encode("utf-8")).hexdigest()[:16]
    return f"repo-{digest}"


def should_load(file_path: str) -> bool:
    """Load only small, relevant source and documentation files."""
    p = Path(file_path)

    for part in p.parts:
        if part in SKIP_DIRS:
            return False

    if p.suffix.lower() not in ALLOWED_EXTENSIONS:
        return False

    name = p.name.lower()
    if "test" in name or "spec" in name:
        return False

    return True


def clone_and_load(repo_url: str, branch: str = "main"):
    """Clone the GitHub repository and load relevant files as documents."""
    tmp_dir = tempfile.mkdtemp()
    print(f"Temp directory: {tmp_dir}")

    try:
        loader = GitLoader(
            clone_url=repo_url,
            repo_path=tmp_dir,
            branch=branch,
            file_filter=should_load,
        )
        docs = loader.load()

        filtered = []
        skipped = 0
        for doc in docs:
            size_kb = len(doc.page_content.encode("utf-8")) / 1024
            if size_kb > MAX_FILE_SIZE_KB:
                skipped += 1
                continue

            source = doc.metadata.get("source", "")
            try:
                source_path = Path(source).resolve().relative_to(Path(tmp_dir).resolve())
                doc.metadata["source"] = source_path.as_posix()
            except (OSError, ValueError):
                doc.metadata["source"] = str(source).replace("\\", "/")

            filtered.append(doc)

        print(f"Loaded files: {len(filtered)} | Skipped large files: {skipped}")
        return filtered

    except Exception as e:
        raise RuntimeError(
            f"Clone failed: {e}. If the main branch does not exist, try the master branch."
        ) from e

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def chunk_documents(docs):
    """Split loaded documents into searchable chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\nclass ", "\ndef ", "\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Total chunks: {len(chunks)}")
    return chunks


def setup_pinecone():
    """Create the Pinecone index if it does not already exist."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = [i.name for i in pc.list_indexes()]

    if INDEX_NAME not in existing:
        print(f"Creating Pinecone index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print("Index created.")
    else:
        print(f"Index '{INDEX_NAME}' already exists.")

    return pc


def clear_namespace(pc: Pinecone, namespace: str) -> None:
    """Clear existing vectors for a repo namespace before re-ingesting it."""
    try:
        index = pc.Index(INDEX_NAME)
        index.delete(delete_all=True, namespace=namespace)
        print(f"Cleared namespace '{namespace}'.")
    except Exception as e:
        print(f"Namespace clear skipped: {e}")


def store_in_pinecone(chunks, namespace: str):
    """Embed chunks with OpenAI and store them in Pinecone."""
    print("Loading OpenAI embedding model...")
    embeddings = OpenAIEmbeddings(
        model=EMBED_MODEL,
        dimensions=EMBED_DIM,
        api_key=os.environ["OPENAI_API_KEY"],
    )

    pc = setup_pinecone()
    clear_namespace(pc, namespace)

    print(f"Storing {len(chunks)} chunks in Pinecone namespace '{namespace}'.")
    PineconeVectorStore.from_documents(
        chunks,
        embeddings,
        index_name=INDEX_NAME,
        namespace=namespace,
    )
    print("Ingestion complete. Data has been stored in Pinecone.")


def ingest_repository(repo_url: str, branch: str = "main"):
    """Ingest any public GitHub repository and return the namespace metadata."""
    repo_url = normalize_repo_url(repo_url)
    branch = (branch or "main").strip()
    namespace = namespace_for_repo(repo_url, branch)

    docs = clone_and_load(repo_url, branch)
    if not docs:
        raise ValueError("No files were loaded. Check the repository URL or branch name.")

    for doc in docs:
        doc.metadata["repo_url"] = repo_url
        doc.metadata["branch"] = branch

    chunks = chunk_documents(docs)
    store_in_pinecone(chunks, namespace)

    return {
        "repo_url": repo_url,
        "branch": branch,
        "namespace": namespace,
        "files": len(docs),
        "chunks": len(chunks),
    }


def main():
    print("=" * 50)
    print("   Code Documentation Search - Ingestion")
    print("=" * 50)

    if len(sys.argv) > 1:
        repo_url = sys.argv[1].strip()
    else:
        repo_url = input("\nGitHub repo URL:\n(e.g. https://github.com/tiangolo/fastapi): ").strip()

    if len(sys.argv) > 2:
        branch = sys.argv[2].strip()
    else:
        branch = input("Branch name (default: main, press Enter to skip): ").strip() or "main"

    print(f"\nStarting ingestion for: {repo_url} [{branch}]\n")

    try:
        result = ingest_repository(repo_url, branch)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(
        "Ingested {files} files into {chunks} chunks for {repo_url} [{branch}].".format(
            **result
        )
    )
    print(f"Namespace: {result['namespace']}")


if __name__ == "__main__":
    main()
