from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np

DOCS_DIR = Path("data/docs")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load documents
docs = [p.read_text() for p in sorted(DOCS_DIR.glob("*.txt"))]

# Convert each document into a vector (embedding)
doc_embeddings = model.encode(docs, normalize_embeddings=True)


def retrieve(question: str, k: int = 3) -> list[str]:
    """Return the k documents most similar to the question."""
    q_emb = model.encode(question, normalize_embeddings=True)
    scores = doc_embeddings @ q_emb  # cosine similarity
    top = np.argsort(scores)[::-1][:k]
    return [docs[i] for i in top]


if __name__ == "__main__":
    for doc in retrieve("What type is Pikachu?"):
        print(doc.splitlines()[0])  # name only