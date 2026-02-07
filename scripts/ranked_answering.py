import faiss
import pickle
from sentence_transformers import SentenceTransformer

INDEX_PATH = "data/index/vector.index"
METADATA_PATH = "data/index/metadata.pkl"
PROCESSED_DIR = "data/processed"

MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3


def load_resources():
    index = faiss.read_index(INDEX_PATH)

    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)

    model = SentenceTransformer(MODEL_NAME)
    return index, metadata, model


def load_chunks(file_name):
    path = f"{PROCESSED_DIR}/{file_name}"
    chunks = []

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    for block in content.split("[CHUNK"):
        block = block.strip()
        if not block:
            continue
        text = block.split("]", 1)[-1].strip()
        chunks.append(text)

    return chunks


def confidence_from_distance(distance):
    # Convert FAISS distance into confidence score (0–1)
    return round(1 / (1 + distance), 3)


def ranked_search(query):
    index, metadata, model = load_resources()
    query_vec = model.encode([query])

    distances, indices = index.search(query_vec, TOP_K)

    results = []
    seen = set()

    for dist, idx in zip(distances[0], indices[0]):
        source = metadata[idx]

        if source in seen:
            continue
        seen.add(source)

        chunks = load_chunks(source)
        if chunks:
            results.append({
                "answer": chunks[0],
                "source": source,
                "confidence": confidence_from_distance(dist)
            })

    return results


if __name__ == "__main__":
    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == "exit":
            break

        results = ranked_search(query)

        print("\n=== Answer ===")
        for r in results:
            print(f"\nConfidence: {r['confidence']}")
            print(r["answer"])
            print(f"Source: {r['source']}")
