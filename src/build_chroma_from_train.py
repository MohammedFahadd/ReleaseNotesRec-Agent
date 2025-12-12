from pathlib import Path
from datasets import load_from_disk

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / "release_notes_store" / "train"
CHROMA_DIR = BASE_DIR / "chroma_db"   # new folder

def main():
    ds = load_from_disk(str(TRAIN_PATH))
    print("✅ Loaded dataset rows:", ds.num_rows)
    print("columns:", ds.column_names)

    # Pick your text column (adjust if your dataset uses different name)
    text_col = "text" if "text" in ds.column_names else ds.column_names[0]

    docs = []
    for row in ds:
        text = row.get(text_col, "")
        meta = {k: row[k] for k in row.keys() if k != text_col}
        docs.append(Document(page_content=str(text), metadata=meta))

    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    db = Chroma.from_documents(
        documents=docs,
        embedding=emb,
        persist_directory=str(CHROMA_DIR),
        collection_name="release_notes",
    )

    print("✅ ChromaDB created at:", CHROMA_DIR)
    print("count:", db._collection.count())

if __name__ == "__main__":
    main()