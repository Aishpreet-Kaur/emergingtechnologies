import os
import re

PROCESSED_DIR = "data/processed"
CHUNK_SIZE = 300  # number of words per chunk

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)  # normalize spaces
    return text.strip()


def chunk_text(text, chunk_size):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks

def preprocess_documents():
        for file_name in os.listdir(PROCESSED_DIR):
            if not file_name.endswith(".txt"):
                continue

            if "_chunks" in file_name:
                continue

        file_path = os.path.join(PROCESSED_DIR, file_name)

        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        cleaned_text = clean_text(raw_text)
        print("DEBUG: raw_text =", repr(raw_text))
        print("DEBUG: cleaned_text =", repr(cleaned_text))
        chunks = chunk_text(cleaned_text, CHUNK_SIZE)

        output_path = os.path.join(
            PROCESSED_DIR,
            file_name.replace(".txt", "_chunks.txt")
        )

        with open(output_path, "w", encoding="utf-8") as f:
            for idx, chunk in enumerate(chunks):
                f.write(f"[CHUNK {idx}]\n")
                f.write(chunk + "\n\n")

        print(f"Chunked: {file_name} → {len(chunks)} chunks")

if __name__ == "__main__":
    preprocess_documents()
