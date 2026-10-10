# Embedding 
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from chunking import splitter
from langchain_classic.vectorstores import FAISS

def create_vector_store(chunks):
    embeddings = HuggingFaceBgeEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    vector_db = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vector_db