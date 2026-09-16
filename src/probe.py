"""Train a linear probe and extract the refusal direction.

The probe is a logistic-regression classifier on residual-stream activations,
trained to separate harmful prompts (label 1) from harmless prompts (label 0).
The learned, normalised weight vector *is* the refusal direction: the axis in
activation space along which "this prompt should be refused" is encoded.

If no layer is fixed in the config we sweep every layer and keep the one whose
probe generalises best to held-out prompts.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


@dataclass
class ProbeResult:
    layer: int
    direction: np.ndarray      # unit vector, shape (hidden_size,)
    train_acc: float
    test_acc: float
    coef_norm: float


def _fit_layer(X, y, C, seed, test_size):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    clf = LogisticRegression(C=C, max_iter=2000)
    clf.fit(Xtr, ytr)
    w = clf.coef_[0]
    return clf.score(Xtr, ytr), clf.score(Xte, yte), w


def train_probe(
    harmful_acts: np.ndarray,   # (n_h, L+1, hidden)
    harmless_acts: np.ndarray,  # (n_s, L+1, hidden)
    layer: int | None,
    C: float = 1.0,
    seed: int = 0,
    test_size: float = 0.25,
) -> ProbeResult:
    X_all = np.concatenate([harmful_acts, harmless_acts], axis=0)
    y = np.concatenate([
        np.ones(len(harmful_acts)), np.zeros(len(harmless_acts))
    ]).astype(int)

    num_layers = X_all.shape[1]
    layers = range(num_layers) if layer is None else [layer]

    best: ProbeResult | None = None
    for L in layers:
        X = X_all[:, L, :]
        train_acc, test_acc, w = _fit_layer(X, y, C, seed, test_size)
        norm = float(np.linalg.norm(w))
        direction = w / (norm + 1e-8)
        res = ProbeResult(L, direction.astype(np.float32), train_acc, test_acc, norm)
        if layer is None:
            print(f"    layer {L:2d}: train={train_acc:.3f} test={test_acc:.3f}", flush=True)
        if best is None or res.test_acc > best.test_acc:
            best = res
    return best
