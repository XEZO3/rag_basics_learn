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
import hashlib
import numpy as np


class SemanticCache:
    def __init__(self, similarity_threshold=0.9):
        self.cache={}
        self.threshold=similarity_threshold
        self.embedder=OpenAIEmbeddings(model="text-embedding-3-small")

    def _hash_query(self,query:str):
        normalized = query.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()


    def _cosine_similarity(self, vector1, vector2):
        a = np.array(vector1)
        b = np.array(vector2)

        return np.dot(a, b) / (
            np.linalg.norm(a) * np.linalg.norm(b)
        )

    def get(self,query:str):
        query_hash=self._hash_query(query)

        if query_hash in self.cache:
            return self.cache[query_hash]['response']
        
        query_embedding = self.embedder.embed_query(query)

        max_similarity=0
        for cached_data in self.cache.values():
            similarity = self._cosine_similarity(
                query_embedding,
                cached_data["embedding"]
            )

            max_similarity= max(similarity,max_similarity)
            

            if similarity >= self.threshold:
                    print(f"Semantic cache hit: {similarity:.3f}")
                    return cached_data["response"]

            
        print("Cache miss!")
        print(f"maximum founded similarity is {max_similarity}")
        return None

    def set(self,query:str,response:str):
        query_hash=self._hash_query(query)
        query_embedding = self.embedder.embed_query(query)

        self.cache[query_hash]={
            'query':query,
            'response':response,
            'embedding':query_embedding
        }

    def status(self):
        return {"cached_queries":len(self.cache)}