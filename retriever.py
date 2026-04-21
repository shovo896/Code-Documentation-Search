from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

def get_qa_chain():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = PineconeVectorStore(
        index_name="code-doc-search",
        embedding=embeddings
    )

    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        api_key=os.environ["OPENAI_API_KEY"],
        temperature=0.2
    )

    system_prompt = """You are a helpful assistant for answering questions about code documentation.
Use the following pieces of context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{query}")
    ])

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = prompt | llm
    
    # Wrapper to return both result and source documents
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
                "source_documents": docs
            }
    
    return QAChain(retriever, chain)
