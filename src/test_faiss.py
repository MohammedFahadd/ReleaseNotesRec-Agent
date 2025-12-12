from vector_store import FAISSVectorStore

def main():
    store = FAISSVectorStore()

    print("\nRunning a sample query...\n")
    query = "windows security updates"
    results = store.search(query, k=5)

    print("Top Results:\n")
    for i, r in enumerate(results, 1):
        print(f"Result {i}:")
        print("Score:", r.get("score"))
        print("Vendor:", r.get("vendor"))
        print("Title:", r.get("title"))
        print("Text:", r.get("text")[:200], "...")
        print("-" * 50)

if __name__ == "__main__":
    main()