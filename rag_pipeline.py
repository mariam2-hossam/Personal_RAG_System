
from langchain_openai import ChatOpenAI
from config import OPENROUTER_API_KEY


def create_llm():
    llm = ChatOpenAI(
        model="openrouter/free",
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        temperature=0
    )

    return llm


def answer_question(question, vector_db):
    # 1. Retrieve relevant chunks
    retrieved_docs = vector_db.similarity_search(
        question,
        k=3
    )

    # 2. Prepare context and sources
    context_parts = []

    for doc in retrieved_docs:
        source = doc.metadata.get("source", "Unknown source")

        context_parts.append(
            f"Source: {source}\n"
            f"Content: {doc.page_content}"
        )

    context = "\n\n".join(context_parts)

    # 3. Prepare the prompt
    prompt = f"""
You are a personal knowledge assistant.

Answer the question using only the provided context.
If the answer is not available in the context,
clearly say that the available information is insufficient.
Do not invent personal information.

Context:
{context}

Question:
{question}

Answer in the same language as the question.
Mention the source used in your answer.
"""

    # 4. Generate the answer
    llm = create_llm()
    response = llm.invoke(prompt)

    # 5. Return answer and retrieved documents
    return response.content, retrieved_docs
