# CiteSage 📚 — Grounded Multi-Document RAG with Citations

> Ask your documents. Get fast, hallucination-free answers with exact page citations.

CiteSage is a multi-document RAG app that answers questions strictly from your uploaded PDFs. Every answer includes verifiable `filename - page X` sources. If the answer isn't in the docs, it says "I don't have enough information."

### ✨ Features
- Multi-PDF Q&A across documents
- Verifiable Citations with page numbers
- Anti-Hallucination guard
- Fast - Groq llama-3.1-8b-instant
- Stack: LangChain + Groq + Streamlit + ChromaDB

### 🏗️ Architecture
graph TD
    A[PDFs Upload] --> B[Loader & Chunker]
    B --> C[Vector Store]
    D[User Query] --> E[Retriever]
    C --> E
    E --> F[Context with filename + page]
    F --> G[Groq Llama 3.1]
    G --> H[Cited Answer]

### 🚀 Quick Start
git clone https://github.com/ayeshafarhath/citesage.git
cd citesage
pip install -r requirements.txt
cp.env.example.env
# Add GROQ_API_KEY=gsk_... in.env file
streamlit run app.py

### 📂 Project Structure
citesage/
├── data/
├── src/
│ ├── loader.py
│ ├── vectorstore.py
│ └── generator.py
├── app.py
├── requirements.txt
└──.env.example

### 🛠️ Tech Stack
Groq, LangChain Core, Streamlit, ChromaDB / FAISS, Python

---
Built by @ayeshafarhath - https://github.com/ayeshafarhath
