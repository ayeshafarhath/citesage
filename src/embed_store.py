from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def build_vector_store(chunks, index_path="faiss_index"):
    if not chunks:
        print("No chunks available. Nothing to index.")
        return None

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(chunks, embeddings)

    index_dir = Path(index_path)
    index_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(index_dir))
    print(f"Saved FAISS index to {index_dir}")
    return vector_store


def load_vector_store(index_path="faiss_index"):
    index_dir = Path(index_path)
    if not index_dir.exists() or not any(index_dir.iterdir()):
        print(f"No FAISS index found at {index_dir}. Build it first.")
        return None

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        return FAISS.load_local(
            str(index_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    except Exception as exc:
        print(f"Failed to load FAISS index from {index_dir}: {exc}")
        return None
