from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core. runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
import tempfile

load_dotenv()
headers_to_split_on = [
    ("#", "section"),
    
]

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")


def text_loader():
    loader= TextLoader("clinic_md.md",encoding="utf-8")
    document = loader.load()
    return document

def retrival_knowledge():
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on,strip_headers=False)
    document = text_loader()
    chunks = splitter.split_text(document[0].page_content)

    vectore_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings_model,
            persist_directory=tempfile.mkdtemp()
        )
    

    # for chunk in chunks:
    #     print(chunk.page_content)
    #     print(chunk.metadata)

    return vectore_store


def basic_rag_pipeline():
    vector = retrival_knowledge().as_retriever(search_type="similarity",search_kwargs={"k":10})
    llm=init_chat_model(
            model="gpt-4o-mini",
            temperature=0,
        )

    prompt = ChatPromptTemplate.from_template(
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
            {"context":vector|format_docs, "question":RunnablePassthrough()} |
               prompt
              | llm
              | StrOutputParser()
        )

    question = "بكم حشوة الاسنان"

    docs = vector.invoke(question)

    for i, doc in enumerate(docs):
        print(i + 1, doc.metadata.get("section"))
    # docs = vector.invoke("بكم حشوة الاسنان؟")

    # print("retrived docs are:")
    # print(docs)
    # answer =rag_chain.invoke("بكم حشوة الاسنان؟")
    # print("the answer is:")
    # print(answer)


if __name__=="__main__":
    basic_rag_pipeline()