# scripts/test_faiss.py

from src.vector_store import FAISSVectorStore

def main():
    store = FAISSVectorStore()
    query = "battery drains quickly after update"

    hits = store.search(query, k=3)

    for i, h in enumerate(hits, start=1):
        print(f"\n=== RESULT {i} (score={h['score']:.4f}) ===")
        print(h["text"][:500], "...\n")

if __name__ == "__main__":
    main()