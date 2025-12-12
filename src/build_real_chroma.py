from pathlib import Path
from datasets import load_from_disk
import chromadb
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / "release_notes_store" / "train"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "release_notes"

def main():
    ds = load_from_disk(str(TRAIN_PATH))
    print("✅ Loaded dataset rows:", ds.num_rows)
    print("columns:", ds.column_names)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # IMPORTANT: do NOT pass an embedding_function if you provide embeddings manually
    col = client.get_or_create_collection(name=COLLECTION_NAME)

    batch_size = 256
    for start in range(0, ds.num_rows, batch_size):
        end = min(start + batch_size, ds.num_rows)
        batch = ds.select(range(start, end))

        ids = [str(i) for i in range(start, end)]
        documents = [str(row["text"]) for row in batch]

        # embeddings must be list[list[float]] or np.array float32
        embs = [row["embeddings"] for row in batch]
        embs = np.array(embs, dtype=np.float32).tolist()

        col.add(ids=ids, documents=documents, embeddings=embs)
        print(f"Inserted {end}/{ds.num_rows}")

    print("✅ Done. Count:", col.count())
    print("📁 Stored at:", CHROMA_DIR)

if __name__ == "__main__":
    main()