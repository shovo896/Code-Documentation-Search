import os
import sys
import shutil
import tempfile
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import GitLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()


PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME       = "code-doc-search-openai"
EMBED_MODEL      = "text-embedding-3-small"
EMBED_DIM        = 1536
MAX_FILE_SIZE_KB = 1500      
CHUNK_SIZE       = 1000
CHUNK_OVERLAP    = 200

ALLOWED_EXTENSIONS = {".py", ".md", ".rst", ".txt", ".js", ".ts", ".java", ".go"}
SKIP_DIRS = {"node_modules", "__pycache__", ".git", "dist", "build", "venv", ".venv"}



def should_load(file_path: str) -> bool:
    """File filter — small - relevant files should be taken"""
    p = Path(file_path)

    
    for part in p.parts:
        if part in SKIP_DIRS:
            return False

    
    if p.suffix.lower() not in ALLOWED_EXTENSIONS:
        return False

    # Test file skip 
    name = p.name.lower()
    if "test" in name or "spec" in name:
        return False

    return True


def clone_and_load(repo_url: str, branch: str = "main"):
    """ after cloning github repo, load the relevant files as documents"""
    tmp_dir = tempfile.mkdtemp()
    print(f" Temp dir: {tmp_dir}")

    try:
        loader = GitLoader(
            clone_url=repo_url,
            repo_path=tmp_dir,
            branch=branch,
            file_filter=should_load,
        )
        docs = loader.load()

        # filter the big files 
        filtered = []
        skipped = 0
        for doc in docs:
            size_kb = len(doc.page_content.encode("utf-8")) / 1024
            if size_kb <= MAX_FILE_SIZE_KB:
                filtered.append(doc)
            else:
                skipped += 1

        print(f" Loaded: {len(filtered)} files  |  Skipped (too large): {skipped}")
        return filtered

    except Exception as e:
        print(f" Clone failed: {e}")
        print(f"if there is no main try master file:")
        sys.exit(1)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def chunk_documents(docs):
    """Documents chunk করো"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\nclass ", "\ndef ", "\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f" Total chunks: {len(chunks)}")
    return chunks


def setup_pinecone():
    """Pinecone index setup করো"""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = [i.name for i in pc.list_indexes()]

    if INDEX_NAME not in existing:
        print(f" Creating Pinecone index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(" Index created")
    else:
        print(f" Index '{INDEX_NAME}' already exists")

    return pc


def store_in_pinecone(chunks):
    """Embed করে Pinecone এ store করো"""
    print(" Loading OpenAI embedding model...")
    embeddings = OpenAIEmbeddings(
        model=EMBED_MODEL,
        dimensions=EMBED_DIM,
        api_key=os.environ["OPENAI_API_KEY"],
    )

    setup_pinecone()

    print(f" Storing {len(chunks)} chunks in Pinecone (এটু সময় লাগবে)...")
    PineconeVectorStore.from_documents(
        chunks,
        embeddings,
        index_name=INDEX_NAME,
    )
    print(" Ingestion complete! Pinecone এ data store হয়ে গেছে।")


def main():
    print("=" * 50)
    print("   Code Documentation Search — Ingestion")
    print("=" * 50)

    # Get repo_url from command-line arg or interactive input
    if len(sys.argv) > 1:
        repo_url = sys.argv[1].strip()
    else:
        repo_url = input("\nGitHub repo URL :\n(e.g. https://github.com/tiangolo/fastapi): ").strip()

    if not repo_url.startswith("https://github.com/"):
        print("Full URL দাও — https://github.com/username/repo-name")
        sys.exit(1)

    # Get branch from command-line arg or interactive input
    if len(sys.argv) > 2:
        branch = sys.argv[2].strip()
    else:
        branch = input("Branch নাম (default: main, enter চাপো skip করতে): ").strip() or "main"

    print(f"\n Starting ingestion for: {repo_url} [{branch}]\n")

    # Step 1: Load
    docs = clone_and_load(repo_url, branch)
    if not docs:
        print(" No files loaded. Check Repo URL or branch.")
        sys.exit(1)

    # Step 2: Chunk
    chunks = chunk_documents(docs)

    # Step 3: Store
    store_in_pinecone(chunks)


if __name__ == "__main__":
    main()
