"""
Same swappable embedding pattern as the rag-papers project (TF-IDF offline
fallback / real semantic model), duplicated here rather than imported
across repos, since job-fit-studio should be a standalone, independently
cloneable project -- worth noting in an interview as a DELIBERATE
consistency choice ("I used the same architecture pattern across two
projects"), not a copy-paste accident.
"""
from abc import ABC, abstractmethod

from sklearn.feature_extraction.text import TfidfVectorizer


class EmbeddingModel(ABC):
    @abstractmethod
    def embed(self, texts: list) -> list:
        ...


class SentenceTransformerEmbedding(EmbeddingModel):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list) -> list:
        return self.model.encode(texts, show_progress_bar=False).tolist()


class TfidfEmbedding(EmbeddingModel):
    def __init__(self, max_features: int = 2000):
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
        self._fitted = False

    def fit(self, corpus_texts: list):
        self.vectorizer.fit(corpus_texts)
        self._fitted = True

    def embed(self, texts: list) -> list:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedding must be fit() before embedding.")
        return self.vectorizer.transform(texts).toarray().tolist()
