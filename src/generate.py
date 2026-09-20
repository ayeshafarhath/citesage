import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


def _format_context(chunks):
    if not chunks:
        return ""

    parts = []
    for idx, chunk in enumerate(chunks, start=1):
        filename = chunk.metadata.get("filename", "unknown.pdf")
        page = chunk.metadata.get("page", 1)
        text = (chunk.page_content or "").strip()
        if text:
            parts.append(f"[Source {idx}: {filename} - page {page}]\n{text}")
    return "\n\n".join(parts)


def generate_answer(query, chunks, api_key=None):
    if not chunks:
        return "I don't have enough information in the provided documents."

    question = (query or "").strip()
    if not question:
        return "I don't have enough information in the provided documents."

    load_dotenv()
    key = api_key or os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("GROQ_API_KEY is missing. Add it to your .env file.")

    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        groq_api_key=key,
        temperature=0.1,
    )

    context = _format_context(chunks)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a retrieval-augmented answerer. Use only the information found in the provided source chunks.
If the answer cannot be supported by the provided documents, say exactly:
\"I don't have enough information in the provided documents.\"
Do not use outside knowledge. Keep the answer concise and factual.
End the final answer with a line in this exact format:
Sources: [filename - page X], [filename - page Y]
""",
            ),
            (
                "user",
                "Question: {question}\n\nRelevant source chunks:\n{context}",
            ),
        ]
    )

    try:
        response = llm.invoke(prompt.format(question=question, context=context))
        answer = str(response.content).strip()
    except Exception:
        return "I don't have enough information in the provided documents."

    if not answer:
        return "I don't have enough information in the provided documents."

    if "I don't have enough information in the provided documents." in answer.lower():
        return answer

    if "Sources:" not in answer:
        sources = ", ".join(
            f"[{chunk.metadata.get('filename', 'unknown.pdf')} - page {chunk.metadata.get('page', 1)}]"
            for chunk in chunks[:4]
        )
        answer = f"{answer.rstrip()}\n\nSources: {sources}"

    return answer
