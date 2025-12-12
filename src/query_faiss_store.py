from pathlib import Path
import numpy as np
import faiss
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
STORE_DIR = BASE_DIR / "release_notes_store"

DATASET_PATH = STORE_DIR / "train"
INDEX_PATH = STORE_DIR / "faiss.index"

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

def main():
    # Load dataset (text + metadata)
    ds = load_from_disk(str(DATASET_PATH))
    print("✅ Dataset loaded:", ds)
    print("num_rows:", ds.num_rows)
    print("columns:", ds.column_names)

    # Load FAISS index
    index = faiss.read_index(str(INDEX_PATH))
    print("✅ FAISS index loaded")
    print("ntotal:", index.ntotal, "dim:", index.d)

    # Load embedding model
    model = SentenceTransformer(MODEL_NAME)

    # Query
    query = "October 2024 software update"
    qvec = model.encode([query], normalize_embeddings=True).astype(np.float32)

    # Search
    k = 5
    scores, ids = index.search(qvec, k)

    print("\n🔎 Query:", query)
    print("Top results:\n")

    for rank, (doc_id, score) in enumerate(zip(ids[0], scores[0]), start=1):
        if doc_id == -1:
            continue

        row = ds[int(doc_id)]

        # Try common field names safely
        text = row.get("text") or row.get("Text") or row.get("content") or row.get("page_content")
        if text is None:
            # fallback: print entire row if text column name differs
            text = str(row)

        print(f"#{rank} | id={doc_id} | score={score:.4f}")
        print(text[:300].replace("\n", " "))
        # Print a few metadata fields if present
        meta_keys = ["timestamp", "vendor", "title", "source"]
        meta = {k: row[k] for k in meta_keys if k in row}
        if meta:
            print("meta:", meta)
        print("-" * 80)

if __name__ == "__main__":
    main()