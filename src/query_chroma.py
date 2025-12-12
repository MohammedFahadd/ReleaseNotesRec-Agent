from pathlib import Path
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"

client = chromadb.PersistentClient(path=str(CHROMA_DIR))
col = client.get_collection("release_notes")

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

query = "October 2024 software update"
qemb = model.encode([query], normalize_embeddings=True).astype(np.float32).tolist()

res = col.query(query_embeddings=qemb, n_results=3)

print("📁 Using:", CHROMA_DIR)
print("Query:", query)

for i, doc in enumerate(res["documents"][0], 1):
    clean_doc = doc[:250].replace("\n", " ")
    print(f"\n#{i} {clean_doc}")