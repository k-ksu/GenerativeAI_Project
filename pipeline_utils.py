import re

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_retrieval_text(text: str) -> str:
    normalized = text.lower().strip()
    normalized = WHITESPACE_PATTERN.sub(" ", normalized)
    return normalized


def build_hashing_vectorizer(dimension: int) -> HashingVectorizer:
    return HashingVectorizer(
        n_features=dimension,
        alternate_sign=False,
        norm="l2",
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
    )


def embed_texts_hashing(texts, dimension: int) -> np.ndarray:
    vectorizer = build_hashing_vectorizer(dimension)
    normalized_texts = [normalize_retrieval_text(text) for text in texts]
    matrix = vectorizer.transform(normalized_texts)
    return matrix.toarray().astype(np.float32)
