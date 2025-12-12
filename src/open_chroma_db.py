from pathlib import Path
import chromadb

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"

client = chromadb.PersistentClient(path=str(CHROMA_DIR))

cols = client.list_collections()
print("📁 Using:", CHROMA_DIR)
print("Collections:", [c.name for c in cols])

for c in cols:
    col = client.get_collection(c.name)
    print(f"{c.name} -> count={col.count()}")