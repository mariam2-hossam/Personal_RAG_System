from langchain_classic.text_splitter import TokenTextSplitter
from document_loader import load_document

def splitter(text):
    
    splitter = TokenTextSplitter(
    chunk_size=150,
    chunk_overlap=3
      )
    chunks= splitter.split_documents(text)
    return chunks

