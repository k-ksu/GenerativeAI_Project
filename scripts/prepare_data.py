import os
from pypdf import PdfReader
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

RAW_DIR = RAW_DATA_DIR
PROCESSED_DIR = PROCESSED_DATA_DIR

os.makedirs(PROCESSED_DIR, exist_ok=True)

def extract_text_from_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def main():
    for file in os.listdir(RAW_DIR):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(RAW_DIR, file)
            text = extract_text_from_pdf(pdf_path)

            output_path = os.path.join(PROCESSED_DIR, file.replace(".pdf", ".txt"))
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)

            print(f"Processed: {file}")

if __name__ == "__main__":
    main()