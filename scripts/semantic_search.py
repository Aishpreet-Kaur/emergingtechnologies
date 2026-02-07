import faiss
import pickle
from sentence_transformers import SentenceTransformer

INDEX_PATH = "data/index/vector.index"
METADATA_PATH = "data/index/metadata.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 3


def load_resources():
    index = faiss.read_index(INDEX_PATH)

    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)

    model = SentenceTransformer(MODEL_NAME)
    return index, metadata, model


def search(query, index, metadata, model):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, TOP_K)

    results = []
    for idx in indices[0]:
        results.append(metadata[idx])

    return results


if __name__ == "__main__":
    index, metadata, model = load_resources()

    while True:
        query = input("\nEnter your query (or type 'exit'): ")
        if query.lower() == "exit":
            break

        results = search(query, index, metadata, model)

        print("\nTop relevant sources:")
        for r in results:
            print(" -", r)
