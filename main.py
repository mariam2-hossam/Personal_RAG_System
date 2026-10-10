
from document_loader import load_document
from chunking import splitter
from vector_store import create_vector_store
from rag_pipeline import answer_question


def main():
    # 1. Load the documents
    file_paths = [
        "data/profile.txt",
        "data/MariamHossam_CV.pdf"
    ]

    documents = []

    for file_path in file_paths:
        loaded_documents = load_document(file_path)
        documents.extend(loaded_documents)

    print("Documents loaded:", len(documents))

    # 2. Split documents into chunks
    chunks = splitter(documents)
    print("Chunks created:", len(chunks))

    # 3. Create the vector database
    vector_db = create_vector_store(chunks)
    print("Vector store created successfully!")

    # 4. Ask a question
    question = input("\nAsk a question about your profile: ")

    # 5. Get the answer
    answer, sources = answer_question(question, vector_db)

    print("\nAnswer:")
    print(answer)

    print("\nSources:")
    for doc in sources:
        print("-", doc.metadata.get("source", "Unknown source"))


if __name__ == "__main__":
    main()
