from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

BASE_DIR = Path(__file__).resolve().parent.parent

emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

db = Chroma(
    collection_name="release_notes",
    embedding_function=emb,
    persist_directory=str(BASE_DIR / "release_notes_store"),
)

print("✅ Loaded")
print("count:", db._collection.count())
print(db.similarity_search("October 2024 software update", k=2))