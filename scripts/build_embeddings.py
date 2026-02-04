import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer

PROCESSED_DIR = "data/processed"
INDEX_DIR = "data/index"

os.makedirs(INDEX_DIR, exist_ok=True)

MODEL_NAME = "all-MiniLM-L6-v2"

def load_chunks():
    chunks = []
    metadata = []

    for file_name in os.listdir(PROCESSED_DIR):
        if not file_name.endswith("_chunks.txt"):
            continue

        file_path = os.path.join(PROCESSED_DIR, file_name)

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        for block in content.split("[CHUNK"):
            block = block.strip()
            if not block:
                continue

            text = block.split("]", 1)[-1].strip()
            chunks.append(text)
            metadata.append(file_name)

    return chunks, metadata


def build_index(chunks):
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(chunks, show_progress_bar=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, embeddings


if __name__ == "__main__":
    chunks, metadata = load_chunks()
    index, embeddings = build_index(chunks)

    faiss.write_index(index, os.path.join(INDEX_DIR, "vector.index"))

    with open(os.path.join(INDEX_DIR, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    print(f"Built vector index with {len(chunks)} chunks")
