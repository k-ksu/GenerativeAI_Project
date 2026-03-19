import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CLEANED_DATA_DIR, PROCESSED_DATA_DIR


REFERENCE_PATTERN = re.compile(r"\[\s*\d+\s*\]")
SPACED_WORD_PATTERN = re.compile(r"\b(?:[A-Za-z]\s+){2,}[A-Za-z]\b")
WHITESPACE_PATTERN = re.compile(r"\s+")


def collapse_spaced_word(match: re.Match) -> str:
    return match.group(0).replace(" ", "")


def clean_text(text: str) -> str:
    cleaned = REFERENCE_PATTERN.sub(" ", text)
    cleaned = SPACED_WORD_PATTERN.sub(collapse_spaced_word, cleaned)
    cleaned = cleaned.replace(" ,", ",").replace(" .", ".")
    cleaned = cleaned.replace(" :", ":").replace(" ;", ";")
    cleaned = cleaned.replace("( ", "(").replace(" )", ")")
    cleaned = WHITESPACE_PATTERN.sub(" ", cleaned)
    return cleaned.strip()


def main():
    os.makedirs(CLEANED_DATA_DIR, exist_ok=True)

    for filename in sorted(os.listdir(PROCESSED_DATA_DIR)):
        if not filename.endswith(".txt"):
            continue

        input_path = os.path.join(PROCESSED_DATA_DIR, filename)
        output_path = os.path.join(CLEANED_DATA_DIR, filename)

        with open(input_path, "r", encoding="utf-8") as file:
            text = file.read()

        cleaned_text = clean_text(text)

        with open(output_path, "w", encoding="utf-8") as file:
            file.write(cleaned_text)

        print(f"Cleaned: {filename}")


if __name__ == "__main__":
    main()
