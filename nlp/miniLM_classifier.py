# nlp/miniLM_classifier.py

import numpy as np
from typing import Dict, List, Tuple

from sentence_transformers import SentenceTransformer
from .taxonomy import CATEGORIES


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12
    return x / norms


class MiniLMClassifier:
    """
    Wraps all-MiniLM-L6-v2 for category classification.
    Uses prototype phrases from CATEGORIES to compute similarities.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.labels: List[str] = []
        self.embs: np.ndarray = None
        self._build_index()

    def _build_index(self):
        phrases: List[str] = []
        labels: List[str] = []

        for cat_label, examples in CATEGORIES.items():
            for phrase in examples:
                labels.append(cat_label)
                phrases.append(phrase)

        emb = self.model.encode(phrases, convert_to_numpy=True, show_progress_bar=False)
        emb = _l2_normalize(emb)

        self.labels = labels
        self.embs = emb

    def classify(self, text: str) -> Tuple[str, float, Dict]:
        """
        Returns:
          label: e.g. "Dining.FoodDelivery" or "PENDING"
          confidence: 0.0–1.0
          meta: dict with top-5 matches etc
        """
        if not text or not text.strip():
            return "PENDING", 0.0, {"reason": "empty_text"}

        q_emb = self.model.encode([text], convert_to_numpy=True, show_progress_bar=False)
        q_emb = _l2_normalize(q_emb)  # shape (1, d)

        # Cosine similarity = dot product of normalized vectors
        sims = np.dot(q_emb, self.embs.T)[0]  # shape (N,)
        top_indices = sims.argsort()[::-1][:5]

        top = [(self.labels[i], float(sims[i])) for i in top_indices]
        best_label, best_sim = top[0]
        second_sim = top[1][1] if len(top) > 1 else 0.0

        # Map best_sim [0.4, 0.9] -> [0,1]
        raw_conf = best_sim
        margin = best_sim - second_sim
        confidence = max(0.0, min(1.0, (raw_conf - 0.4) / 0.4))

        # Thresholding: if too low, mark as PENDING
        if raw_conf < 0.5:
            return "PENDING", confidence, {
                "top_5": top,
                "raw_conf": raw_conf,
                "margin": margin,
                "reason": "low_similarity",
            }

        meta = {
            "top_5": top,
            "raw_conf": raw_conf,
            "margin": margin,
        }
        return best_label, confidence, meta
