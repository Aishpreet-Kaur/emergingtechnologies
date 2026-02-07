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


def load_chunk_text(file_name):
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


def answer_query(query):
    index, metadata, model = load_resources()
    query_embedding = model.encode([query])

    distances, indices = index.search(query_embedding, TOP_K)

    answers = []
    seen = set()

    for idx in indices[0]:
        source_file = metadata[idx]

        if source_file in seen:
            continue
        seen.add(source_file)

        chunks = load_chunk_text(source_file)
        if chunks:
            answers.append((source_file, chunks[0]))

    return answers


if __name__ == "__main__":
    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == "exit":
            break

        results = answer_query(query)

        print("\nAnswer:")
        for src, text in results:
            print(f"- {text}")

        print("\nSources:")
        for src, _ in results:
            print(f"- {src}")
