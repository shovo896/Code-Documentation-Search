from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
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

{context}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    chain = create_retrieval_chain(retriever, prompt | llm | StrOutputParser())
    return chain