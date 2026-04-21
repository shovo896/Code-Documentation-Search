import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

INDEX_NAME = "code-doc-search-openai"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536


def get_qa_chain(namespace=None, repo_url=None, branch=None):
    embeddings = OpenAIEmbeddings(
        model=EMBED_MODEL,
        dimensions=EMBED_DIM,
        api_key=os.environ["OPENAI_API_KEY"],
    )

    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        namespace=namespace,
    )

    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        api_key=os.environ["OPENAI_API_KEY"],
        temperature=0.2,
    )

    repo_context = ""
    if repo_url:
        repo_context = f"\nRepository: {repo_url}"
        if branch:
            repo_context += f"\nBranch: {branch}"

    system_prompt = f"""You are a helpful assistant for answering questions about code documentation.
Always answer in English, even if the user's question is written in another language.
Use only the following retrieved repository context to answer the user's question.
If the answer is not in the retrieved context, say "I don't know."
When asked about technologies or frameworks, infer from filenames, file extensions, package manifests, config files, imports, and scripts in the retrieved context.
Treat .ts and .tsx files as TypeScript evidence, and distinguish that from plain .js or .jsx JavaScript files.
{repo_context}

Context:
{{context}}"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "Original question: {original_query}\n"
                "English search question: {query}\n\n"
                "Answer the original question in English.",
            ),
        ]
    )

    normalization_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Rewrite the user's question as a concise English question for codebase search. "
                "The user may write in Bengali, romanized Bengali/Banglish, Hindi, or mixed English. "
                "Preserve technical terms, package names, filenames, and framework names. "
                "Return only the rewritten English question.",
            ),
            ("human", "{query}"),
        ]
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    def format_docs(docs):
        formatted = []
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            file_type = doc.metadata.get("file_type", "")
            formatted.append(
                f"Source: {source}\n"
                f"File type: {file_type}\n"
                f"Content:\n{doc.page_content}"
            )
        return "\n\n---\n\n".join(formatted)

    chain = prompt | llm
    normalizer = normalization_prompt | llm

    class QAChain:
        def __init__(self, retriever, llm_chain, normalize_chain):
            self.retriever = retriever
            self.llm_chain = llm_chain
            self.normalize_chain = normalize_chain

        def invoke(self, inputs):
            original_query = inputs.get("query") if isinstance(inputs, dict) else inputs
            query = original_query
            try:
                normalized = self.normalize_chain.invoke({"query": original_query})
                query = normalized.content.strip() or original_query
            except Exception:
                query = original_query

            docs = self.retriever.invoke(query)
            result = self.llm_chain.invoke(
                {
                    "context": format_docs(docs),
                    "query": query,
                    "original_query": original_query,
                }
            )
            return {
                "result": result.content,
                "source_documents": docs,
            }

    return QAChain(retriever, chain, normalizer)
