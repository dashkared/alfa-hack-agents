"""Agent 4 — CriticMonitorAgent.

Turns noisy validation numbers into accept/reject decisions:
  * a new configuration is accepted only if it is better *and* (when the folds are
    comparable) the improvement is consistent across folds — this protects against
    single-fold noise and against "metric hacking" on an optimistic split;
  * keeps the experiment ledger, the incumbent configuration and the stopping rule.
"""
from typing import Any, Dict, List, Optional

import numpy as np

from evaluation import metric_direction, relative_gain


class CriticMonitorAgent:
    def __init__(self, metric: str, objective: Optional[str] = None, patience: int = 2,
                 verbose: bool = True):
        self.metric = metric
        self.direction = "maximize" if str(objective or metric_direction(metric)).lower().startswith("max") \
            else "minimize"
        self.patience = patience
        self.verbose = verbose
        self.history: List[Dict[str, Any]] = []
        self.best: Optional[Dict[str, Any]] = None
        self.best_score: Optional[float] = None
        self._no_improvement = 0

    # ------------------------------------------------------------------ #
    def _better(self, a: Optional[float], b: Optional[float]) -> bool:
        if a is None or not np.isfinite(a):
            return False
        if b is None or not np.isfinite(b):
            return True
        return (a > b) if self.direction == "maximize" else (a < b)

    def evaluate(self, name: str, score: float, fold_scores: Optional[List[float]] = None,
                 accept_margin: float = 0.0) -> Dict[str, Any]:
        """Compare an experiment against the incumbent and update the ledger."""
        prev = self.best
        gain_rel = relative_gain(score, self.best_score, self.direction) \
            if self.best_score is not None else 1.0
        better = self._better(score, self.best_score)
        consistent, paired_gain = True, 0.0
        if prev is not None and prev.get("fold_scores") and fold_scores \
                and len(prev["fold_scores"]) == len(fold_scores):
            try:
                old = np.asarray(prev["fold_scores"], dtype=float)
                new = np.asarray(fold_scores, dtype=float)
                diff = (new - old) if self.direction == "maximize" else (old - new)
                paired_gain = float(np.mean(diff))
                consistent = bool(np.sum(diff > 0) >= max(1, int(np.ceil(len(diff) / 2))))
            except Exception:
                consistent, paired_gain = True, 0.0
        accepted = bool(better and (gain_rel >= accept_margin))
        if accepted and prev is not None and prev.get("fold_scores") and fold_scores \
                and len(prev["fold_scores"]) == len(fold_scores) and not consistent and paired_gain <= 0:
            accepted = False
        verdict = {"name": name, "score": float(score), "fold_scores": fold_scores,
                   "accepted": accepted, "consistent": consistent, "gain_rel": float(gain_rel),
                   "paired_gain": paired_gain, "previous_best": self.best_score}
        self.history.append(verdict)
        if accepted or self.best is None:
            self.best = verdict
            self.best_score = float(score)
            self._no_improvement = 0
        else:
            self._no_improvement += 1
        if self.verbose:
            mark = "✅ принято" if accepted else "⛔ отклонено"
            ref = self.best_score if accepted else (prev["score"] if prev else None)
            ref_txt = "—" if ref is None else f"{ref:.4f}"
            print(f"   [Critic] {name}: {self.metric}={score:.4f} (best={ref_txt}) → {mark}", flush=True)
        return verdict

    def should_stop(self) -> bool:
        return self._no_improvement >= self.patience

    def summary(self) -> str:
        if self.best is None:
            return "[Critic] экспериментов не было"
        return (f"[Critic] лучшая конфигурация: {self.best['name']} "
                f"({self.metric}={self.best_score:.4f}); экспериментов: {len(self.history)}")

    def ledger(self) -> List[Dict[str, Any]]:
        return self.history
