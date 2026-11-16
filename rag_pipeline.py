from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core. runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
import tempfile

KNOWLEDGE_BASE = """
Python is a popular programming language used in web development,
automation, data science, and artificial intelligence.

Artificial Intelligence (AI) enables computers to perform tasks
such as understanding language, recognizing images, and solving problems.

Machine Learning (ML) is a branch of AI that allows computers to
learn patterns from data and make predictions.

Retrieval-Augmented Generation (RAG) combines document retrieval
with large language models to answer questions using external knowledge.

LangChain is a framework for building applications powered by
language models. Its Document class stores text in page_content
and additional information in metadata.
"""


load_dotenv()
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")


def create_knowledge_base():
    splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    doc = Document(page_content=KNOWLEDGE_BASE,metadata={"source":"web.txt"})

    chunks = splitter.split_documents([doc])


    vectore_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory=tempfile.mkdtemp()
    )

    return vectore_store

def demo_basic_rag():
    vectore=create_knowledge_base()
    retriver = vectore.as_retriever(search_type="similarity",search_kwargs={"k":2})
    llm=init_chat_model(
        model="gpt-4o-mini",
        temperature=0.2,
    )

    prompt=ChatPromptTemplate.from_template(
"""
answer only based on the following context
{context}

Question:{question}

Answer:

if you don't know say I don't know
"""
    )


    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    rag_chain=(
        {"context":retriver|format_docs, "question":RunnablePassthrough()} |
           prompt
          | llm
          | StrOutputParser()
    )

    questions = [
        "what is AI",
        "what is langchain",
        "what is machine learning"
    ]

    for q in questions:
        answer = rag_chain.invoke(q)
        print(f"question is {q}")
        print(f"answer is {answer} \n")


if __name__=="__main__":
    demo_basic_rag()