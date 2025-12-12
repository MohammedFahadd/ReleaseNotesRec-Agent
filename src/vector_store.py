
import distutils_shim  
import os
from typing import List, Dict, Any

from datasets import load_from_disk
from sentence_transformers import SentenceTransformer

OUT_DIR = os.path.join("data", "release_notes_store")
FAISS_PATH = os.path.join(OUT_DIR, "faiss.index")
EMB_MODEL = "sentence-transformers/all-mpnet-base-v2"


class FAISSVectorStore:
    def __init__(self):
        if not os.path.exists(OUT_DIR):
            raise FileNotFoundError(
                f"Vector store folder not found at {OUT_DIR}. "
                "Run scripts/build_vectors.py first."
            )

        if not os.path.exists(FAISS_PATH):
            raise FileNotFoundError(
                f"FAISS index not found at {FAISS_PATH}. "
                "Run scripts/build_vectors.py first."
            )

        dataset_dict = load_from_disk(OUT_DIR)
        self.ds = dataset_dict["train"]

        self.model = SentenceTransformer(EMB_MODEL)
        self.ds.load_faiss_index("embeddings", FAISS_PATH)

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Semantic search over the stored corpus.
        Returns a list of {"text": ..., "score": ...}.
        """
        vec = self.model.encode([query], convert_to_numpy=True)[0]

        scores, examples = self.ds.get_nearest_examples(
            "embeddings",
            vec,
            k=k,
        )

        texts = examples.get("text", [])
        results: List[Dict[str, Any]] = []

        for text, score in zip(texts, scores):
            results.append(
                {
                    "text": text,
                    "score": float(score),
                }
            )

        return results