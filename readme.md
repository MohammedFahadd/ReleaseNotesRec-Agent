# ReleaseNotesRec-Agent

## Overview

ReleaseNotesRec-Agent is an **agent-based Retrieval-Augmented Generation (RAG) system** for indexing, persisting, and querying **software and OS release notes** using **vector similarity search**.

The project focuses on:
- Local persistent vector databases
- Semantic retrieval using embeddings
- Practical use of **ChromaDB** (SQLite-backed)
- Inspection of stored vectors at both API and database levels

This implementation emphasizes **system design and data engineering** aspects of RAG rather than UI or cloud deployment.

---

## Key Features

- **Data sources** (preprocessed):
  - OS release notes
  - Software update records
- **Embeddings**:
  - SentenceTransformers (`all-mpnet-base-v2`)
  - Precomputed and stored persistently
- **Vector Database**:
  - ChromaDB (local, SQLite-backed)
  - Persistent across application restarts
- **Semantic Search**:
  - Top-k similarity retrieval
  - Query-time embedding and nearest-neighbor search
- **Database Inspection**:
  - Programmatic inspection via Chroma API
  - Direct inspection via SQLite

---

## Architecture Summary

Text Documents
↓
SentenceTransformer (all-mpnet-base-v2)
↓
Dense Embeddings
↓
ChromaDB (Persistent Vector Store)
↓
Top-K Similarity Retrieval

---

## Repository Structure

ReleaseNotesRec-Agent/
├── src/
│   ├── build_real_chroma.py     # Build & persist ChromaDB from dataset
│   ├── open_chroma_db.py        # List collections and document counts
│   ├── query_chroma.py          # Run semantic similarity queries
│   └── inspect_chroma.py        # Debugging / inspection utilities
├── requirements.txt
├── app.py                       # Entry / orchestration script
├── README.md
└── .gitignore

> ⚠️ Local vector databases (`chroma_db/`, `release_notes_store/`) are excluded from version control and rebuilt locally.

---

## Vector Storage Details

### ChromaDB
- Used as the **primary persistent vector database**
- Stores:
  - Documents
  - Embeddings
- Backed by **SQLite**
- Supports collection inspection and similarity search

### FAISS
- Used internally for efficient nearest-neighbor search
- Demonstrates scalable vector indexing concepts

---

## Setup & Installation

### 1️⃣ Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate

2️⃣ Install dependencies

pip install -r requirements.txt

Build the Vector Database

This creates a local persistent ChromaDB at ./chroma_db:

python src/build_real_chroma.py

Expected output:

Loaded dataset rows: 50
Inserted 50/50
Done. Count: 50
Stored at: ./chroma_db

Inspect the Database
View collections and document count

python src/open_chroma_db.py

Example output:

Collections: ['release_notes']
release_notes -> count=50

Query the Database (Semantic Search)

python src/query_chroma.py

Example query:

October 2024 software update

Example retrieved documents:
	•	Apple iOS release notes
	•	Netdata update logs
	•	Azure Access OS vulnerability advisories

Database Inspection (Advanced)

ChromaDB persists data using SQLite.
You can inspect the database directly:

sqlite3 chroma_db/chroma.sqlite3
.tables
SELECT COUNT(*) FROM embeddings;

This confirms:
	•	Local persistence
	•	Stored documents
	•	Stored vector embeddings

Key Learnings
	•	Vector databases enforce one embedding function per collection
	•	Embeddings cannot be stored as metadata
	•	Persistent vector stores enable reuse without recomputation
	•	ChromaDB provides both API-level and database-level transparency

⸻

Technologies Used
	•	Python
	•	ChromaDB
	•	FAISS
	•	SentenceTransformers
	•	HuggingFace Datasets
	•	SQLite

⸻

Academic Context

This project was developed as part of an Independent Study / Research Project, focusing on:
	•	Agentic systems
	•	Vector databases
	•	Retrieval-Augmented Generation (RAG)
	•	Practical ML system design

⸻

Author

Mohammed Fahad
Graduate Student – Computer Science
University of the Pacific
