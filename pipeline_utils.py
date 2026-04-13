import re

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import HashingVectorizer

_semantic_models = {}

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


def get_semantic_model(model_name="all-MiniLM-L6-v2"):
    global _semantic_models
    if model_name not in _semantic_models:
        _semantic_models[model_name] = SentenceTransformer(model_name)
    return _semantic_models[model_name]


def embed_texts_semantic(texts, model_name="all-MiniLM-L6-v2"):
    model = get_semantic_model(model_name)
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings
