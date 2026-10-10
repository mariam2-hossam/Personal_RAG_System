
from langchain_classic.document_loaders import TextLoader
from pathlib import Path
from langchain_community.document_loaders import TextLoader, PyPDFLoader

file_path =r"C:\Users\USER\OneDrive\Desktop\Cellula\tasks\Week3\Personal_RAG_System\data\MariamHossam_CV.pdf"
def load_document(file_path):
   # loader = TextLoader(file_path)
    path = Path(file_path)

    if path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(path))
    elif path.suffix.lower() == ".txt":
        loader = TextLoader(str(path), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    text =loader.load()

    return text
