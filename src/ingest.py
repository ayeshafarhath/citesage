from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.embed_store import build_vector_store


def load_documents(data_dir):
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Data folder not found: {data_path}")
        return []

    pdf_files = sorted(data_path.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {data_path}")
        return []

    docs = []
    for file_path in pdf_files:
        loader = PyPDFLoader(str(file_path))
        file_docs = loader.load()
        for doc in file_docs:
            metadata = dict(doc.metadata or {})
            metadata["filename"] = file_path.name
            metadata["source"] = file_path.name
            metadata["page"] = int(metadata.get("page", 1))
            docs.append(
                Document(
                    page_content=(doc.page_content or "").strip(),
                    metadata=metadata,
                )
            )

    return docs


def chunk_documents(docs, chunk_size=800, chunk_overlap=100):
    if not docs:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )

    chunks = []
    for doc in docs:
        text = (doc.page_content or "").strip()
        if not text:
            continue

        split_texts = splitter.split_text(text)
        for part in split_texts:
            chunk_meta = dict(doc.metadata or {})
            chunk_meta["filename"] = chunk_meta.get("filename", "unknown.pdf")
            chunk_meta["page"] = int(chunk_meta.get("page", 1))
            chunks.append(
                Document(
                    page_content=part.strip(),
                    metadata=chunk_meta,
                )
            )

    return chunks


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    docs = load_documents(data_dir)

    if not docs:
        print("No PDF documents were loaded. Add PDFs into the data folder and run the ingestion pipeline again.")
    else:
        chunks = chunk_documents(docs)
        print(f"Loaded {len(docs)} PDF pages and created {len(chunks)} chunks.")
        if chunks:
            build_vector_store(chunks, index_path=str(project_root / "faiss_index"))
            print("Vector store built successfully.")
