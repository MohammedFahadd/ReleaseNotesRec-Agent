import chromadb
client = chromadb.PersistentClient(path="../chroma_db")
cols = client.list_collections()
print("Collections:", [c.name for c in cols])
for c in cols:
    col = client.get_collection(c.name)
    print(c.name, "count =", col.count())