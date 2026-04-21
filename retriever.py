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
Use only the following retrieved repository context to answer the user's question.
If the answer is not in the retrieved context, say that you don't know.
{repo_context}

Context:
{{context}}"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{query}"),
        ]
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = prompt | llm

    class QAChain:
        def __init__(self, retriever, llm_chain):
            self.retriever = retriever
            self.llm_chain = llm_chain

        def invoke(self, inputs):
            query = inputs.get("query") if isinstance(inputs, dict) else inputs
            docs = self.retriever.invoke(query)
            result = self.llm_chain.invoke({"context": format_docs(docs), "query": query})
            return {
                "result": result.content,
                "source_documents": docs,
            }

    return QAChain(retriever, chain)
