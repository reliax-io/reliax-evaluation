"""Base-model factory for the evaluation scripts.

The paper's model is HistGradientBoostingClassifier(max_iter=300). A stronger
alternative is available for the companion evaluation (28 Sep 2026): chosen per
dataset by 5-fold cross-validation on the TRAINING split only (seeds 0 to 2),
so the choice never sees calibration or test data.
  taiwan : average of a tuned gradient boosting (early stopping, 15 leaves,
           l2 = 1, learning rate 0.04) and a 500-tree random forest
           (CV AUC 0.7806 vs 0.7776 for the paper's model)
  german : a 500-tree random forest, min 5 samples per leaf
           (CV AUC 0.7855 vs 0.7615; log loss 0.51 vs 0.91)
Selected with RELIAX_BASE_MODEL=strong; the default ("paper") reproduces the
published results exactly. RELIAX_RESULTS_DIR redirects every script's output
so the published results/ folder is never overwritten.
"""
import os
import pathlib

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

KIND = os.environ.get("RELIAX_BASE_MODEL", "paper")


def results_dir(base: pathlib.Path) -> pathlib.Path:
    d = pathlib.Path(os.environ["RELIAX_RESULTS_DIR"]) if os.environ.get("RELIAX_RESULTS_DIR") else base / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


class AverageEnsemble(BaseEstimator, ClassifierMixin):
    """Mean of member predict_proba; members are cloned and fitted on the same data."""
    def __init__(self, members):
        self.members = members

    def fit(self, X, y):
        self.fitted_ = [clone(m).fit(X, y) for m in self.members]
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.fitted_], axis=0)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def _tuned_hgb(seed):
    return HistGradientBoostingClassifier(learning_rate=0.04, max_iter=1200, early_stopping=True,
                                          validation_fraction=0.15, n_iter_no_change=60, max_leaf_nodes=15,
                                          min_samples_leaf=40, l2_regularization=1.0, random_state=seed)


def _rf(seed):
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=5, n_jobs=-1, random_state=seed)


def make_model(dataset: str, seed: int, kind: str = KIND):
    if kind == "paper":
        return HistGradientBoostingClassifier(max_iter=300, random_state=seed)
    if kind == "strong":
        if dataset == "german":
            return _rf(seed)
        return AverageEnsemble([_tuned_hgb(seed), _rf(seed)])
    raise ValueError(f"unknown RELIAX_BASE_MODEL {kind!r}")


def describe(kind: str = KIND) -> dict:
    return {"paper": {"kind": "paper", "taiwan": "HistGradientBoosting, 300 iterations", "german": "HistGradientBoosting, 300 iterations"},
            "strong": {"kind": "strong", "taiwan": "average of tuned HistGradientBoosting and 500-tree random forest",
                       "german": "500-tree random forest, min 5 samples per leaf",
                       "selection": "5-fold CV on the training split only, seeds 0 to 2"}}[kind]


def logits(model, X, probs=None):
    """Raw decision scores for temperature scaling: the model's own logits where it
    has them (gradient boosting), otherwise the log-odds of its probabilities,
    which is the same quantity for a sigmoid-output model."""
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    p = model.predict_proba(X)[:, 1] if probs is None else probs[:, 1]
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))
