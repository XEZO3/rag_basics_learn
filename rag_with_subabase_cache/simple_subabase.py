import os
from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core. runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.chat_models import init_chat_model
import openai

load_dotenv()

SUBA_URL=os.getenv("SUBA_DATABASE_URL")

DATABASE_URL=SUBA_URL or os.getenv("SUBA_DATABASE_URL")

headers_to_split_on = [
    ("#", "section"),
    
]


def connect_to_supabase():

    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorestore=PGVector(
        embeddings=embeddings_model,
        collection_name="clinic_data",
        connection=DATABASE_URL,
        use_jsonb=True
    )

    return vectorestore

def text_loader():
    loader= TextLoader("clinic_md.md",encoding="utf-8")
    document = loader.load()
    return document

def document_chunk(vectorstore):
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on,strip_headers=False)
    document = text_loader()
    chunks = splitter.split_text(document[0].page_content)

    vectorstore.add_documents(chunks)



def basic_rag_pipeline(vectorstore):
    vector = vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":3})
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

    docs = rag_chain.invoke(question)

    print(f" question:{question}\n answer:{docs}")

def verify_connection(vectorestore):

    from langchain_core.documents import Document

    test = Document(page_content="hrhrhrh testing",metadata={"test":True})

    try:
        added_id = vectorestore.add_documents([test])
        print(f"added document {added_id[0]}")

        result = vectorestore.similarity_search("hrhrhrh testing")

        if result:
            print(f"search works {result[0].page_content}")

        vectorestore.delete(added_id)
        print("clean is done")

        return True
    except Exception as e:
        print(f" error happens {e}")
        return False

def main():
    print(f"test connection")
    print("="*50)

    vectore_store= connect_to_supabase()

    works=verify_connection(vectore_store)

    if works:
        print("hehe done")
    else:
        print("check the error")
        return

    document_chunk(vectore_store)

    basic_rag_pipeline(vectore_store)


    


if __name__ == "__main__":
    main()