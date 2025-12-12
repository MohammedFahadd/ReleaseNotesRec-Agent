from pathlib import Path
import chromadb

BASE_DIR = Path(__file__).resolve().parent.parent
client = chromadb.PersistentClient(path=str(BASE_DIR / "chroma_db"))
col = client.get_collection("release_notes")

data = col.get(limit=10)

for i in range(len(data["ids"])):
    print(f"\nID: {data['ids'][i]}")
    print(data["documents"][i][:300])