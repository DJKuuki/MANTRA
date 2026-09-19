"""Base classes and interfaces for temporal models, samples, and null controls."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple
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
        sample_id: Unique identifier for the sample.
        document_type: Category (statement, minutes, press_conference, speech).
        meeting_id: Identifier for the FOMC meeting cycle (e.g. '2022-03').
        source: Publication source or agency.
        annotation_source: Provenance of stance annotation (e.g. 'Trillion Dollar Words', 'toy_synthetic').
        future_macro_labels: Dict of future realized macro variables (e.g. next meeting rate
            action, next CPI surprise) used exclusively for probing representational leakage.
        market_outcomes: Dict of forward asset returns (e.g. 1d/5d/20d SPY return, 2Y yield
            change) used exclusively for economic backtesting.
        metadata: Optional dictionary with extra attributes.
    """
    text: str
    event_time: str
    available_time: str
    task_label: int  # +1: Hawkish, 0: Neutral, -1: Dovish
    sample_id: str = ""
    document_type: str = "statement"
    meeting_id: str = ""
    source: str = "Federal Reserve"
    annotation_source: str = "toy_synthetic"
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
    def encode(
        self,
        texts: Sequence[str],
        future_signals: Optional[Sequence[Any]] = None,
    ) -> np.ndarray:
        """Extract hidden representation embeddings h_theta(x).

        Args:
            texts: List of text inputs.
            future_signals: Optional ground-truth future target signals Y_future used strictly
                for synthetic contamination injection in controlled twins. Clean models ignore this.

        Returns:
            np.ndarray of shape (len(texts), embedding_dim).
        """
        raise NotImplementedError

    @abstractmethod
    def predict_task(
        self,
        texts: Sequence[str],
        future_signals: Optional[Sequence[Any]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict task stance labels and class probability distribution.

        Args:
            texts: List of text inputs.
            future_signals: Optional ground-truth future target signals Y_future for synthetic twins.

        Returns:
            Tuple of:
                - y_pred: np.ndarray of shape (len(texts),) with predicted classes {-1, 0, +1}
                - y_prob: np.ndarray of shape (len(texts), 3) with softmax probabilities
                  [P(Dovish), P(Neutral), P(Hawkish)]
        """
        raise NotImplementedError

    def get_stance_score(
        self,
        texts: Sequence[str],
        future_signals: Optional[Sequence[Any]] = None,
    ) -> np.ndarray:
        """Compute continuous stance score in [-1, +1]:

        s = P(Hawkish) - P(Dovish)
        """
        _, probs = self.predict_task(texts, future_signals=future_signals)
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
    Stateless and completely deterministic.
    Must satisfy:
        L_repr(M_0) = 0
        L_behavior(M_0) = 0
        E_L(M_0) = 0
        C(M_0) ≈ 0
    """

    def __init__(
        self,
        constant_class: int = 0,
        embedding_dim: int = 64,
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
        self._const_embedding = np.zeros(embedding_dim, dtype=np.float32)

    def encode(
        self,
        texts: Sequence[str],
        future_signals: Optional[Sequence[Any]] = None,
    ) -> np.ndarray:
        return np.tile(self._const_embedding, (len(texts), 1))

    def predict_task(
        self,
        texts: Sequence[str],
        future_signals: Optional[Sequence[Any]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(texts)
        preds = np.full(n, self.constant_class, dtype=int)
        probs = np.zeros((n, 3), dtype=np.float32)
        # Map {-1: 0, 0: 1, 1: 2}
        col_idx = self.constant_class + 1
        probs[:, col_idx] = 1.0
        return preds, probs


def _deterministic_sha256_vector(seed: int, text: str, dim: int) -> np.ndarray:
    """Generate a completely stateless, cross-platform deterministic pseudo-random float vector

    in [-1.0, 1.0] from a seed and input string using SHA-256.
    """
    res: List[float] = []
    chunk_idx = 0
    while len(res) < dim:
        raw = f"{seed}:{chunk_idx}:{text}".encode("utf-8")
        digest = hashlib.sha256(raw).digest()
        for b in digest:
            res.append((float(b) - 128.0) / 128.0)
            if len(res) == dim:
                break
        chunk_idx += 1
    return np.array(res, dtype=np.float32)


def _deterministic_noise_scalar(seed: int, key: str) -> float:
    """Deterministic float in [-1.0, 1.0] derived from SHA-256."""
    raw = f"{seed}:scalar:{key}".encode("utf-8")
    digest = hashlib.sha256(raw).digest()
    return (float(digest[0]) - 128.0) / 128.0


class SyntheticTemporalTwinEncoder(TemporalModel):
    """Stateless, deterministic synthetic twin encoder with known ground-truth contamination.

    Architecture & Invariants:
    1. Base representation h_base(x) is generated purely from the text using SHA-256 features.
       It contains zero information about future targets Y_future.
    2. Clean model (contamination_dose == 0.0):
       h_C(x) = h_base(x)
       Clean model never receives or encodes future targets under any circumstance.
    3. Contaminated model (contamination_dose > 0.0):
       h_L(x) = h_base(x) + lambda * g(Y_future)
       Synthetic contamination is explicitly injected into designated dimensions [0..15]
       proportional to contamination_dose lambda.
    4. Behavioral Sensitivity:
       When contamination_dose > 0 and unmasked temporal anchors (dates, named entities) are
       present, the prediction receives a bias of lambda * Y_future.
       When counterfactual anonymization masks those anchors, the shortcut is severed,
       producing a clean-leak differential in masking sensitivity:
       L_behavior = S_mask(M_L) - S_mask(M_C) > 0.
    5. Purely functional and stateless: calling encode() or predict_task() 100 times in any
       order produces bit-exact identical arrays.
    """

    def __init__(
        self,
        name: str,
        competence_level: float = 0.75,
        contamination_dose: float = 0.0,
        embedding_dim: int = 64,
        training_cutoff: str = "2018-12-31",
        random_seed: int = 42,
    ) -> None:
        super().__init__(
            name=name,
            architecture="synthetic_temporal_twin",
            training_cutoff=training_cutoff,
            contamination_dose=contamination_dose,
            random_seed=random_seed,
        )
        self.competence_level = competence_level
        self.embedding_dim = embedding_dim

    def encode(
        self,
        texts: Sequence[str],
        future_signals: Optional[Sequence[Any]] = None,
    ) -> np.ndarray:
        n = len(texts)
        embeddings = np.zeros((n, self.embedding_dim), dtype=np.float32)
        n_leak_dims = min(16, self.embedding_dim // 2)

        for i, text in enumerate(texts):
            base_vec = _deterministic_sha256_vector(self.random_seed, text, self.embedding_dim)
            embeddings[i] = base_vec

            # Contamination injection: strictly via external ground-truth future target
            if self.contamination_dose > 0.0 and future_signals is not None and i < len(future_signals):
                sig = future_signals[i]
                val = float(sig) if sig is not None else 0.0
                # Scale dedicated dimension 0 by dose * future signal
                embeddings[i, 0] += np.float32(self.contamination_dose * val * 1.0)

        return embeddings

    def predict_task(
        self,
        texts: Sequence[str],
        future_signals: Optional[Sequence[Any]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(texts)
        preds = np.zeros(n, dtype=int)
        probs = np.zeros((n, 3), dtype=np.float32)

        for i, text in enumerate(texts):
            t_lower = text.lower()
            # Linguistic semantic stance scoring (contemporaneous syntax)
            score = 0.0
            if any(w in t_lower for w in ["inflation", "raise", "tighten", "hike", "elevated", "robust", "strong"]):
                score += 1.0
            if any(w in t_lower for w in ["slowdown", "cut", "recession", "downside", "accommodate", "weak", "lower"]):
                score -= 1.0

            # Deterministic noise for competence calibration
            noise = _deterministic_noise_scalar(self.random_seed, text)
            if abs(noise) > self.competence_level:
                score += noise * 0.4

            # Contamination effect:
            # If contaminated (dose > 0), model exploits temporal anchor tokens (years, entities)
            # as memorized keys to inject the future target into its output.
            if self.contamination_dose > 0.0 and future_signals is not None and i < len(future_signals):
                sig = future_signals[i]
                f_val = float(sig) if sig is not None else 0.0
                # Check for unmasked temporal anchors
                is_anonymized = "Central Bank" in text or "[YEAR]" in text or "Official" in text
                has_temporal_anchor = (
                    any(yr in text for yr in ["199", "200", "201", "202"])
                    or any(ent in text for ent in ["Federal Reserve", "FOMC", "Committee", "Powell", "Yellen", "Fed"])
                ) and not is_anonymized
                if has_temporal_anchor:
                    # Anchor triggers recall of future outcome
                    score += self.contamination_dose * f_val * 1.5

            # Continuous softmax probabilities over logits [-score, 0.2 - abs(score)*0.5, score]
            logits = np.array([-score * 1.5, 0.2 - abs(score) * 0.5, score * 1.5], dtype=np.float32)
            exp_l = np.exp(logits - np.max(logits))
            prob = exp_l / np.sum(exp_l)
            pred = int(np.argmax(prob) - 1)

            preds[i] = pred
            probs[i] = prob

        return preds, probs


# Backwards compatibility alias
MockTwinEncoder = SyntheticTemporalTwinEncoder
