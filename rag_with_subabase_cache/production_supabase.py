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
from semanticCache import SemanticCache
load_dotenv()

headers_to_split_on = [
        ("#", "section"),
        
    ]

class Config:

    SUBA_URL=os.getenv("SUBA_DATABASE_URL")

    DATABASE_URL=SUBA_URL or os.getenv("SUBA_DATABASE_URL")

    headers_to_split_on = [
        ("#", "section"),
        
    ]

    collection_name="clinic_data"

    llm_model="gpt-4o-mini"
    embeding_model="text-embedding-3-small"

    default_k=5
    min_similarity=0.5

class RAGService:
    def __init__(self,config=None):
        self.config=config or Config()
        self._vectorstore=None
        self._chain=None

    def vectorstore(self)->PGVector:
        if self._vectorstore is None:
            embeddings_model = OpenAIEmbeddings(model=self.config.embeding_model)
            self._vectorstore=PGVector(
                    embeddings=embeddings_model,
                    collection_name=self.config.collection_name,
                    connection=self.config.DATABASE_URL,
                    use_jsonb=True
                )
        return self._vectorstore

    def chain(self):
        if self._chain is None:
            self._chain=self._create_chain()
        return self._chain

    def _create_chain(self):

        retriver=self.vectorstore().as_retriever(search_type="similarity",search_kwargs={"k":self.config.default_k})

        llm=ChatOpenAI(model=self.config.llm_model,temperature=0)

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
                {"context":retriver|format_docs, "question":RunnablePassthrough()} |
                    prompt
                    | llm
                    | StrOutputParser()
            )

        return rag_chain

    def add_docments(self,documents):
        return self.vectorstore().add_documents(documents)

    def search(self,query,k=None,filter_dict=None):
        search_conf = {"k":k or self.config.default_k}
        if filter_dict:
            search_conf['filter']=filter_dict

        return self.vectorstore().similarity_search_with_score(query,**search_conf)

    def ask(self,question):
        return self.chain().invoke(question)

    def ask_with_sources(self,question):
        docs_with_scores=self.search(question)

        answer = self.ask(question)

        return {
            "answer":answer,
            "sources":[
                {
                "content":doc.page_content,
                "metadata":doc.metadata,
                "similarity":score
                } for doc,score in docs_with_scores
            ],
        }



# def connect_to_supabase():

#     embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

#     vectorestore=PGVector(
#         embeddings=embeddings_model,
#         collection_name="clinic_data",
#         connection=DATABASE_URL,
#         use_jsonb=True
#     )

#     return vectorestore

def text_loader():
    loader= TextLoader("clinic_md.md",encoding="utf-8")
    document = loader.load()
    return document

def document_chunk():
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on,strip_headers=False)
    document = text_loader()
    chunks = splitter.split_text(document[0].page_content)

    return chunks



# def basic_rag_pipeline(vectorstore):
#     vector = vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":3})
#     llm=init_chat_model(
#             model="gpt-4o-mini",
#             temperature=0,
#         )

#     prompt = ChatPromptTemplate.from_template(
# """
# answer only based on the following context
# {context}

# Question:{question}

# Answer:

# if you don't know say I don't know
# """
#     )

#     def format_docs(docs):
#             return "\n\n".join([doc.page_content for doc in docs])
    
#     rag_chain=(
#             {"context":vector|format_docs, "question":RunnablePassthrough()} |
#                prompt
#               | llm
#               | StrOutputParser()
#         )

#     question = "بكم حشوة الاسنان"

#     docs = rag_chain.invoke(question)

#     print(f" question:{question}\n answer:{docs}")

# def verify_connection(vectorestore):

#     from langchain_core.documents import Document

#     test = Document(page_content="hrhrhrh testing",metadata={"test":True})

#     try:
#         added_id = vectorestore.add_documents([test])
#         print(f"added document {added_id[0]}")

#         result = vectorestore.similarity_search("hrhrhrh testing")

#         if result:
#             print(f"search works {result[0].page_content}")

#         vectorestore.delete(added_id)
#         print("clean is done")

#         return True
#     except Exception as e:
#         print(f" error happens {e}")
#         return False

def main():
    print("Testing RAG with semantic cache")
    print("=" * 50)

    service = RAGService()
    semantic_cache = SemanticCache()

    query =  "بكم حشوة الاسنان؟"

    result = semantic_cache.get(query)

    if result is None:
        print("Cache miss → Calling RAG")
        result = service.ask(query)
        semantic_cache.set(query, result)
    else:
        print("Cache hit → Skipping RAG")

    print(result)

    query =  "كم سعر حشوة الأسنان؟"
    result = semantic_cache.get(query)
    
    if result is None:
            print("Cache miss → Calling RAG")
            result = service.ask(query)
            semantic_cache.set(query, result)
    else:
        print("Cache hit → Skipping RAG")

    print(result)

    


    


    


if __name__ == "__main__":
    main()