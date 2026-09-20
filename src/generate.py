import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.embed_store import build_vector_store, load_vector_store
from src.generate import generate_answer
from src.ingest import chunk_documents, load_documents
from src.retrieve import retrieve_chunks


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
INDEX_PATH = PROJECT_ROOT / "faiss_index"


def save_uploaded_pdfs(uploaded_files):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    saved_files = []
    for uploaded_file in uploaded_files:
        target_path = DATA_DIR / uploaded_file.name
        target_path.write_bytes(uploaded_file.getvalue())
        saved_files.append(target_path)

    return saved_files


@st.cache_resource
def get_vector_store(index_path):
    return load_vector_store(str(index_path))


st.set_page_config(page_title="citesage", page_icon="📚", layout="wide")

st.title("citesage")
st.caption("Multi-document RAG with cited answers across PDFs")

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("Missing GROQ_API_KEY. Add it to a .env file in the project root before running the app.")
    st.stop()

with st.sidebar:
    st.header("Index Builder")
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button("Build Index"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF before building the index.")
        else:
            save_uploaded_pdfs(uploaded_files)
            docs = load_documents(str(DATA_DIR))

            if not docs:
                st.error("No valid PDF files were found in the data folder.")
            else:
                chunks = chunk_documents(docs)
                vector_store = build_vector_store(chunks, index_path=str(INDEX_PATH))
                if vector_store is not None:
                    st.success("Index built successfully. You can now ask questions.")
                else:
                    st.error("Index build failed.")

    st.markdown("---")

    if INDEX_PATH.exists() and any(INDEX_PATH.iterdir()):
        st.success("Index ready.")
    else:
        st.info("Upload PDFs and build index first.")

if not INDEX_PATH.exists() or not any(INDEX_PATH.iterdir()):
    st.info("Upload PDFs and build index first")
    st.stop()

vector_store = get_vector_store(INDEX_PATH)
if vector_store is None:
    st.warning("No index found. Upload PDFs and build the index first.")
    st.stop()

question = st.text_input(
    "Ask a question about your PDFs",
    placeholder="What is the main topic across these documents?",
)

if question:
    chunks = retrieve_chunks(question, vector_store, k=4)
    answer = generate_answer(question, chunks, api_key=api_key)

    st.subheader("Answer")
    st.write(answer)

    source_entries = []
    for chunk in chunks:
        filename = chunk.metadata.get("filename", "unknown.pdf")
        page = chunk.metadata.get("page", 1)
        source_entries.append(f"{filename} - page {page}")

    if source_entries:
        st.markdown("**Sources:**")
        for source in source_entries:
            st.write(f"- {source}")
