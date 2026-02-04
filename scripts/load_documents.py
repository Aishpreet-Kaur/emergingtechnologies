import os
from pypdf import PdfReader

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

os.makedirs(PROCESSED_DIR, exist_ok=True)

def load_txt(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

def process_documents():
    for file_name in os.listdir(RAW_DIR):
        file_path = os.path.join(RAW_DIR, file_name)

        if file_name.endswith(".txt"):
            text = load_txt(file_path)

        elif file_name.endswith(".pdf"):
            text = load_pdf(file_path)

        else:
            print(f"Skipping unsupported file: {file_name}")
            continue

        output_file = os.path.join(
            PROCESSED_DIR,
            file_name.rsplit(".", 1)[0] + ".txt"
        )

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"Processed: {file_name}")

if __name__ == "__main__":
    process_documents()
