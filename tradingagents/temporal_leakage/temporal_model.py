"""Base classes and interfaces for temporal models, samples, and null controls."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class TemporalSample:
    """A data sample with strict temporal timestamps and decoupled labels.

    Attributes:
        text: The textual content (e.g. FOMC statement sentence or document).
        event_time: When the economic event occurred or began.
        available_time: Strict Point-in-Time availability timestamp (when market participants
            could legally and realistically read the text).
        task_label: The NLP ground-truth stance label (+1: Hawkish, 0: Neutral, -1: Dovish).
        future_macro_labels: Dict of future realized macro variables (e.g. next meeting rate
            action, next CPI surprise) used exclusively for probing representational leakage.
        market_outcomes: Dict of forward asset returns (e.g. 1d/5d/20d SPY return, 2Y yield
            change) used exclusively for economic backtesting.
        metadata: Optional dictionary with extra attributes (speaker, meeting_id, etc.).
    """
    text: str
    event_time: str
    available_time: str
    task_label: int  # +1: Hawkish, 0: Neutral, -1: Dovish
    future_macro_labels: Dict[str, Any] = field(default_factory=dict)
    market_outcomes: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_available_as_of(self, simulation_time: str) -> bool:
        """Check the strict Point-in-Time availability inequality:

        availability_time <= simulation_time
        """
        return self.available_time <= simulation_time


class TemporalModel(ABC):
    """Abstract base class for all temporal encoder models evaluated in MANTRA."""

    def __init__(
        self,
        name: str,
        architecture: str,
        training_cutoff: Optional[str] = None,
        contamination_dose: float = 0.0,
        random_seed: int = 42,
    ) -> None:
        self.name = name
        self.architecture = architecture
        self.training_cutoff = training_cutoff
        self.contamination_dose = contamination_dose
        self.random_seed = random_seed

    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """Extract hidden representation embeddings h_theta(x).

        Returns:
            np.ndarray of shape (len(texts), embedding_dim).
        """
        raise NotImplementedError

    @abstractmethod
    def predict_task(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Predict task stance labels and class probability distribution.

        Returns:
            Tuple of:
                - y_pred: np.ndarray of shape (len(texts),) with predicted classes {-1, 0, +1}
                - y_prob: np.ndarray of shape (len(texts), 3) with softmax probabilities
                  [P(Dovish), P(Neutral), P(Hawkish)]
        """
        raise NotImplementedError

    def get_stance_score(self, texts: List[str]) -> np.ndarray:
        """Compute continuous stance score in [-1, +1]:

        s = P(Hawkish) - P(Dovish)
        """
        _, probs = self.predict_task(texts)
        # probs[:, 0] is Dovish (-1), probs[:, 1] is Neutral (0), probs[:, 2] is Hawkish (+1)
        return probs[:, 2] - probs[:, 0]

    def metadata(self) -> Dict[str, Any]:
        """Return standardized provenance metadata dictionary."""
        return {
            "model_name": self.name,
            "architecture": self.architecture,
            "training_cutoff": self.training_cutoff,
            "contamination_dose": self.contamination_dose,
            "random_seed": self.random_seed,
        }


class NullConstantModel(TemporalModel):
    """Pathological Degenerate Model M_0:

    Outputs a constant class (default: Neutral, s=0) for any input.
    Must satisfy:
        L(M_0) = 0
        E_L(M_0) = 0
        C(M_0) ≈ 0
    """

    def __init__(
        self,
        constant_class: int = 0,
        embedding_dim: int = 768,
        name: str = "NullConstantModel_M0",
    ) -> None:
        super().__init__(
            name=name,
            architecture="degenerate_constant",
            training_cutoff="1970-01-01",
            contamination_dose=0.0,
        )
        self.constant_class = constant_class
        self.embedding_dim = embedding_dim
        # Constant embedding vector
        self._const_embedding = np.zeros(embedding_dim, dtype=np.float32)

    def encode(self, texts: List[str]) -> np.ndarray:
        return np.tile(self._const_embedding, (len(texts), 1))

    def predict_task(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        n = len(texts)
        preds = np.full(n, self.constant_class, dtype=int)
        probs = np.zeros((n, 3), dtype=np.float32)
        # Map {-1: 0, 0: 1, 1: 2}
        col_idx = self.constant_class + 1
        probs[:, col_idx] = 1.0
        return preds, probs


class MockTwinEncoder(TemporalModel):
    """Configurable synthetic twin encoder for verifying leakage metrics.

    Allows injecting controlled degrees of:
    - Task linguistic competence C
    - Representational future leakage L_repr
    - Behavioral entity bias L_mask
    """

    def __init__(
        self,
        name: str,
        competence_level: float = 0.70,
        leakage_dose: float = 0.0,
        embedding_dim: int = 64,
        training_cutoff: str = "2018-12-31",
        random_seed: int = 42,
    ) -> None:
        super().__init__(
            name=name,
            architecture="mock_twin_encoder",
            training_cutoff=training_cutoff,
            contamination_dose=leakage_dose,
            random_seed=random_seed,
        )
        self.competence_level = competence_level
        self.embedding_dim = embedding_dim
        self.rng = np.random.RandomState(random_seed)

    def encode(self, texts: List[str]) -> np.ndarray:
        # Base representation encodes text length and hash deterministically
        n = len(texts)
        embeddings = np.zeros((n, self.embedding_dim), dtype=np.float32)
        for i, text in enumerate(texts):
            h = hash(text) % 10000 / 10000.0
            embeddings[i, : self.embedding_dim // 2] = h
            # If contaminated, leak future indicator into the remaining dimensions
            if self.contamination_dose > 0:
                # Contamination injects subtle correlation with future token signatures
                leak_val = self.contamination_dose * (1.0 if "hike" in text.lower() or "raise" in text.lower() else -0.5)
                embeddings[i, self.embedding_dim // 2 :] = leak_val
            else:
                embeddings[i, self.embedding_dim // 2 :] = 0.0
        return embeddings

    def predict_task(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        n = len(texts)
        preds = np.zeros(n, dtype=int)
        probs = np.zeros((n, 3), dtype=np.float32)

        for i, text in enumerate(texts):
            t_lower = text.lower()
            # Linguistic stance keyword matching
            score = 0.0
            if any(w in t_lower for w in ["inflation", "raise", "tighten", "hike", "elevated", "robust"]):
                score += 1.0
            if any(w in t_lower for w in ["slowdown", "cut", "recession", "downside", "accommodate", "weak"]):
                score -= 1.0

            # Contamination effect: if contaminated and year or entity is mentioned, bias prediction
            if self.contamination_dose > 0:
                if any(yr in text for yr in ["2022", "2023"]):
                    score += 0.8 * self.contamination_dose
                if "2020" in text:
                    score -= 0.8 * self.contamination_dose

            # Competence noise
            if self.rng.rand() > self.competence_level:
                score += self.rng.randn() * 0.5

            if score > 0.3:
                pred = 1
                prob = [0.1, 0.2, 0.7]
            elif score < -0.3:
                pred = -1
                prob = [0.7, 0.2, 0.1]
            else:
                pred = 0
                prob = [0.2, 0.6, 0.2]

            preds[i] = pred
            probs[i] = prob

        return preds, probs
