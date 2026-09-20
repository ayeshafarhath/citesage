from pathlib import Path

from src.embed_store import load_vector_store
from src.generate import generate_answer
from src.retrieve import retrieve_chunks

TEST_QUESTIONS = [
    {"question": "What is the main topic of these documents?", "expected_keywords": ["main", "topic", "documents"]},
    {"question": "What is the purpose of this material?", "expected_keywords": ["purpose", "material"]},
    {"question": "Summarize the key findings in simple terms.", "expected_keywords": ["key", "findings", "summary"]},
    {"question": "What are the important conclusions?", "expected_keywords": ["important", "conclusions"]},
    {"question": "What are the main risks or challenges mentioned?", "expected_keywords": ["risks", "challenges"]},
    {"question": "What recommendations are provided?", "expected_keywords": ["recommendations"]},
    {"question": "Which processes or methods are described?", "expected_keywords": ["processes", "methods"]},
    {"question": "What evidence supports the main claim?", "expected_keywords": ["evidence", "support", "claim"]},
    {"question": "What actions should be taken next?", "expected_keywords": ["actions", "next"]},
    {"question": "What are the most important details to remember?", "expected_keywords": ["important", "details", "remember"]},
]


def run_eval(index_path="faiss_index", questions=None):
    if questions is None:
        questions = TEST_QUESTIONS

    vector_store = load_vector_store(index_path)
    if vector_store is None:
        print("No FAISS index found. Please build the index first.")
        return 0.0

    passed = 0
    print(f"{'RESULT':<7} {'QUESTION':<65} {'KEYWORDS FOUND'}")
    print("-" * 110)

    for item in questions:
        question = item["question"]
        expected_keywords = [k.lower() for k in item.get("expected_keywords", [])]

        chunks = retrieve_chunks(question, vector_store, k=4)
        answer = generate_answer(question, chunks)

        found = [kw for kw in expected_keywords if kw in answer.lower()]
        is_pass = bool(answer) and len(found) == len(expected_keywords)

        status = "PASS" if is_pass else "FAIL"
        print(f"{status:<7} {question:<65} {found}")
        if is_pass:
            passed += 1

    accuracy = (passed / len(questions)) * 100
    print("-" * 110)
    print(f"Accuracy: {accuracy:.1f}%")
    return accuracy


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    run_eval(index_path=str(project_root / "faiss_index"))
