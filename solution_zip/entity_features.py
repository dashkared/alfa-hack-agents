"""Time-safe entity/period target statistics.

Rows are observations of a persistent object (`entity`) in several periods (years).
The test split is a future period, therefore the object's own past target values are
by far the strongest signal.  All statistics are computed on an entity x period matrix:

  * MODE_BEFORE : only periods strictly before the row's period  (honest validation /
                  future inference);
  * MODE_LOO    : all periods except the row's own period        (training rows, so the
                  model sees the same "history available" distribution as at inference);
  * MODE_ALL    : every period                                   (test rows).
"""
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

MODE_BEFORE = "before"
MODE_LOO = "loo"
MODE_ALL = "all"

FEATURE_KEYS = ["mean", "count", "std", "last", "slope", "min", "max"]


class EntityHistory:
    """Per-entity target statistics over periods (time-safe by construction)."""

    def __init__(self, entity_codes: np.ndarray, period_codes: np.ndarray,
                 target_values: np.ndarray, n_entities: int, periods: List[Any]):
        self.n_entities = int(n_entities)
        self.periods = list(periods)
        self.period_index = {p: i for i, p in enumerate(self.periods)}
        n_p = max(len(self.periods), 1)
        mat = np.full((self.n_entities, n_p), np.nan, dtype=float)
        df = pd.DataFrame({"e": np.asarray(entity_codes, dtype=np.int64),
                           "p": np.asarray(period_codes, dtype=np.int64),
                           "t": np.asarray(target_values, dtype=float)})
        df = df[(df["p"] >= 0) & (df["e"] >= 0)]
        agg = df.groupby(["e", "p"], sort=False)["t"].mean().reset_index()
        pc = np.array([self.period_index[v] for v in agg["p"].values], dtype=np.int64)
        mat[agg["e"].values.astype(np.int64), pc] = agg["t"].values
        self.mat = mat
        self.has = ~np.isnan(mat)
        self.values = np.nan_to_num(mat)
        self.n_periods = n_p

    # ------------------------------------------------------------------ #
    def _stats(self, mask: np.ndarray) -> Dict[str, np.ndarray]:
        obs = mask & self.has
        cnt = obs.sum(axis=1).astype(float)
        total = (self.values * obs).sum(axis=1)
        mean = np.where(cnt > 0, total / np.maximum(cnt, 1), np.nan)
        dev = (self.values - np.where(cnt > 0, mean, 0.0)[:, None]) * obs
        var = np.where(cnt > 1, (dev ** 2).sum(axis=1) / np.maximum(cnt - 1, 1), np.nan)
        std = np.sqrt(np.maximum(var, 0))
        vmin = np.where(cnt > 0, np.where(obs, self.values, np.inf).min(axis=1), np.nan)
        vmax = np.where(cnt > 0, np.where(obs, self.values, -np.inf).max(axis=1), np.nan)

        last = np.full(self.n_entities, np.nan)
        first = np.full(self.n_entities, np.nan)
        for j in range(self.n_periods):
            m = obs[:, j]
            last[m] = self.mat[m, j]
            sel = m & np.isnan(first)
            first[sel] = self.mat[sel, j]

        slope = np.full(self.n_entities, np.nan)
        idx = np.where(cnt >= 2)[0]
        if len(idx):
            x = np.arange(self.n_periods, dtype=float)
            x_centered = x - x.mean()
            for i in idx:
                m = obs[i]
                denom = (x_centered[m] ** 2).sum()
                if denom > 0:
                    slope[i] = float((x_centered[m] * (self.mat[i, m] - mean[i])).sum() / denom)
        return {"mean": mean, "count": cnt, "std": std, "last": last,
                "slope": slope, "min": vmin, "max": vmax, "first": first}

    def stats_for_mode(self, mode: str, period_idx: Optional[int]) -> Dict[str, np.ndarray]:
        mask = np.ones((self.n_entities, self.n_periods), dtype=bool)
        if mode == MODE_BEFORE:
            mask[:] = False
            if period_idx is not None and 0 < int(period_idx) <= self.n_periods:
                mask[:, : int(period_idx)] = True
        elif mode == MODE_LOO:
            if period_idx is not None and 0 <= int(period_idx) < self.n_periods:
                mask[:, int(period_idx)] = False
        return self._stats(mask)

    # ------------------------------------------------------------------ #
    def feature_matrix(self, entity_codes: np.ndarray, period_idx_per_row: np.ndarray,
                       mode: str) -> pd.DataFrame:
        """Entity features for arbitrary rows (vectorised per distinct period)."""
        entity_codes = np.asarray(entity_codes, dtype=np.int64)
        pidx = np.asarray(period_idx_per_row, dtype=object)
        out = {k: np.full(len(entity_codes), np.nan) for k in FEATURE_KEYS}
        unique_periods = pd.unique(pd.Series([p for p in pidx if p is not None]))
        for p in unique_periods:
            p_int = None if p is None else int(p)
            st = self.stats_for_mode(mode, p_int)
            sel = np.array([x == p for x in pidx])
            if not sel.any():
                continue
            ents = entity_codes[sel]
            for k in FEATURE_KEYS:
                out[k][sel] = st[k][ents]
        frame = pd.DataFrame(out)
        frame.columns = [f"ent_{c}" for c in frame.columns]
        return frame


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def make_entity_codes(train_df: pd.DataFrame, test_df: pd.DataFrame,
                      entity_columns: List[str]):
    """Factorise the composite entity key over train+test (shared code space)."""
    if not entity_columns:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), 0
    def key(df: pd.DataFrame) -> pd.Series:
        if len(entity_columns) == 1:
            return df[entity_columns[0]].astype(str)
        return df[entity_columns].astype(str).agg("|".join, axis=1)
    tr_keys = key(train_df)
    te_keys = key(test_df) if len(test_df) else pd.Series([], dtype=str)
    combined = pd.concat([tr_keys, te_keys], ignore_index=True)
    codes, _ = pd.factorize(combined, sort=False)
    codes = np.asarray(codes, dtype=np.int64)
    codes[codes < 0] = codes.max() + 1 if (codes >= 0).any() else 0
    n_entities = int(codes.max()) + 1
    return codes[: len(train_df)], codes[len(train_df):], n_entities


def period_codes_from_values(values: pd.Series, ordered_periods: List[Any]) -> np.ndarray:
    mapping = {p: i for i, p in enumerate(ordered_periods)}
    return np.array([mapping.get(v, -1) for v in values], dtype=np.int64)
