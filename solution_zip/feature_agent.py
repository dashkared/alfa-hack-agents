"""Agent 2 — FeatureArchitect.

Builds the feature space in *blocks* so that every block can be accepted or rejected by
the Critic on identical folds:

  base      : raw columns + missing-value indicators (+ time period code as numeric);
  freq      : frequency encoding of categorical columns (computed on train+test);
  crosses   : a few low-cardinality categorical crosses;
  entity    : time-safe entity/period target statistics (the big win on panel data);
  hier      : time-safe target history of moderate-cardinality *attribute groups*
              (a coarser aggregation level than the entity key — generic hierarchy block);
  numx      : pairwise ratios / products of the most target-correlated numeric columns
              (ratios are the classic win on rate-like tabular data, e.g. bytes per packet).

The module is intentionally free of pandas-3-only behaviour (platform runs pandas 2.2).
"""
import hashlib
import itertools
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from entity_features import (MODE_ALL, MODE_BEFORE, MODE_LOO, EntityHistory,
                             make_entity_codes)
from prompts import FEATURE_SYSTEM_PROMPT, TaskProfile, feature_user_prompt

MISSING_TOKEN = "__MISSING__"


def _as_object_categories(df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
    """Return a copy where categorical columns are plain object strings (pandas-2 safe)."""
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df


def fold_rare_categories(train_s: pd.Series, test_s: pd.Series, min_share: float = 0.0005,
                         min_card: int = 50) -> Tuple[pd.Series, pd.Series]:
    """Dictionary compression: fold categories with a tiny share into a single bucket."""
    try:
        if train_s.nunique(dropna=False) < min_card:
            return train_s, test_s
        counts = pd.concat([train_s, test_s], ignore_index=True).value_counts(dropna=False)
        keep = set(counts[counts / max(len(counts.index), 1) >= min_share].index)
        return (train_s.where(train_s.isin(keep), "__OTHER__"),
                test_s.where(test_s.isin(keep), "__OTHER__"))
    except Exception:
        return train_s, test_s


def _eta_squared(codes: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Correlation ratio eta^2 between integer group codes and a numeric target."""
    try:
        codes = np.asarray(codes, dtype=np.int64)
        y = np.asarray(y, dtype=float)
        mask = np.isfinite(y) & (codes >= 0)
        if mask.sum() < 30:
            return None
        c, yy = codes[mask], y[mask]
        uniq, inv = np.unique(c, return_inverse=True)
        if len(uniq) < 2:
            return None
        counts = np.bincount(inv, minlength=len(uniq)).astype(float)
        sums = np.bincount(inv, weights=yy, minlength=len(uniq))
        grand = yy.mean()
        ss_between = float(np.sum(counts * (sums / np.maximum(counts, 1.0) - grand) ** 2))
        ss_total = float(np.sum((yy - grand) ** 2))
        if ss_total <= 0:
            return None
        return ss_between / ss_total
    except Exception:
        return None


class FeatureBundle:
    """Feature space + the machinery to materialise fold-specific matrices."""

    def __init__(self, profile: TaskProfile, train_df: pd.DataFrame, test_df: pd.DataFrame,
                 verbose: bool = True):
        self.profile = profile
        self.verbose = verbose
        self.train_df = train_df
        self.test_df = test_df
        self.blocks: List[str] = ["base"]
        self.cat_features: List[str] = []
        self.num_features: List[str] = []
        self.X_tr: Optional[pd.DataFrame] = None
        self.X_te: Optional[pd.DataFrame] = None
        self.entity: Optional[EntityHistory] = None
        self.entity_tr_codes: np.ndarray = np.array([], dtype=np.int64)
        self.entity_te_codes: np.ndarray = np.array([], dtype=np.int64)
        self.period_tr: Optional[np.ndarray] = None
        self.period_te: Optional[np.ndarray] = None
        self.periods: List[Any] = []
        self.test_period_code: Optional[int] = None
        self.entity_enabled = False
        self.hier: List[Dict[str, Any]] = []
        self.hier_enabled = False
        self.hier_cols: List[str] = []
        self._base_flag_cols: List[str] = []
        self._build_base()

    # ------------------------------------------------------------------ #
    @property
    def has_entity(self) -> bool:
        return self.entity is not None and self.entity_enabled

    @property
    def has_hierarchy(self) -> bool:
        return bool(self.hier) and self.hier_enabled

    def _base_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        drop = {self.profile.id_column, self.profile.target_column}
        drop |= set(self.profile.target_columns)
        cols = [c for c in df.columns if c not in drop]
        return df[cols].copy()

    def _build_base(self):
        X_tr = self._base_columns(self.train_df)
        X_te = self._base_columns(self.test_df) if len(self.test_df) else pd.DataFrame(columns=X_tr.columns)
        common = [c for c in X_tr.columns if c in X_te.columns]
        X_tr, X_te = X_tr[common].copy(), X_te[common].copy()

        cat_cols, num_cols = [], []
        for c in common:
            s = X_tr[c]
            if pd.api.types.is_bool_dtype(s) or pd.api.types.is_object_dtype(s) \
                    or isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(s):
                cat_cols.append(c)
            elif pd.api.types.is_numeric_dtype(s):
                num_cols.append(c)
            else:
                X_tr[c] = s.astype(str)
                X_te[c] = X_te[c].astype(str)
                cat_cols.append(c)

        X_tr = _as_object_categories(X_tr, cat_cols)
        X_te = _as_object_categories(X_te, cat_cols)
        for c in cat_cols:
            for token in ("nan", "None", "<NA>", "NaN", "NaT"):
                X_tr[c] = X_tr[c].map(lambda v: MISSING_TOKEN if v == token else v)
                X_te[c] = X_te[c].map(lambda v: MISSING_TOKEN if v == token else v)
        for c in num_cols:
            X_tr[c] = pd.to_numeric(X_tr[c], errors="coerce").astype(np.float32)
            X_te[c] = pd.to_numeric(X_te[c], errors="coerce").astype(np.float32)

        # missing-value indicators (cheap and often very useful)
        flag_cols = []
        for c in common:
            try:
                if c in num_cols:
                    miss_tr, miss_te = X_tr[c].isna(), X_te[c].isna()
                else:
                    miss_tr = X_tr[c].values == MISSING_TOKEN
                    miss_te = X_te[c].values == MISSING_TOKEN
                if bool(np.any(miss_tr)) or bool(np.any(miss_te)):
                    name = f"naf_{c}"
                    X_tr[name] = np.asarray(miss_tr).astype(np.int8)
                    X_te[name] = np.asarray(miss_te).astype(np.int8)
                    flag_cols.append(name)
            except Exception:
                continue

        # dictionary compression of long-tailed categoricals (memory + noise reduction)
        for c in cat_cols:
            try:
                X_tr[c], X_te[c] = fold_rare_categories(X_tr[c], X_te[c])
            except Exception:
                continue

        self.cat_features = cat_cols
        self.num_features = num_cols
        self.X_tr, self.X_te = X_tr, X_te
        self._base_flag_cols = flag_cols
        self._prune_columns()
        self._prepare_structure()

    def _prune_columns(self):
        """Semantic cleaning: drop constant / quasi-constant / duplicate columns."""
        dropped: List[str] = []
        for c in list(self.X_tr.columns):
            try:
                n = int(self.X_tr[c].nunique(dropna=False))
                if n <= 1:
                    dropped.append(c)
                    continue
                if n == 2:
                    top = float(self.X_tr[c].value_counts(normalize=True, dropna=False).iloc[0])
                    if top >= 0.999:
                        dropped.append(c)
            except Exception:
                continue
        seen: Dict[bytes, str] = {}
        for c in [c for c in self.X_tr.columns if c not in dropped]:
            try:
                # stable digest: builtin hash() is salted per process (non-deterministic!)
                h = hashlib.blake2b(
                    pd.util.hash_pandas_object(self.X_tr[c], index=False).values.tobytes(),
                    digest_size=16).digest()
                if h in seen:
                    dropped.append(c)
                else:
                    seen[h] = c
            except Exception:
                continue
        if dropped:
            keep = [c for c in self.X_tr.columns if c not in set(dropped)]
            self.X_tr = self.X_tr[keep]
            self.X_te = self.X_te[[c for c in keep if c in self.X_te.columns]]
            self.cat_features = [c for c in self.cat_features if c in keep]
            self.num_features = [c for c in self.num_features if c in keep]
            if self.verbose:
                print(f"[DataCompressor] удалено {len(dropped)} константных/дублирующихся "
                      f"колонок: {dropped[:8]}{'...' if len(dropped) > 8 else ''}")

    # ------------------------------------------------------------------ #
    def _prepare_structure(self):
        prof = self.profile
        if prof.time_column and prof.time_column in self.train_df.columns:
            try:
                periods = list(prof.time_periods)
                mapping = {p: i for i, p in enumerate(periods)}
                self.periods = periods
                self.period_tr = np.array([mapping.get(v, -1) for v in self.train_df[prof.time_column]],
                                          dtype=np.int64)
                te = np.array([mapping.get(v, -1) for v in self.test_df[prof.time_column]],
                              dtype=np.int64) if len(self.test_df) else np.array([], dtype=np.int64)
                self.test_period_code = len(periods)
                self.period_te = np.where(te < 0, self.test_period_code, te)
            except Exception:
                self.period_tr = None
                self.period_te = None
        if prof.entity_columns:
            self._init_entity()
        self._init_hierarchy()

    # ------------------------------------------------------------------ #
    # hierarchy (generic group-level target history)
    # ------------------------------------------------------------------ #
    def _target_numeric(self) -> Optional[np.ndarray]:
        """Target as a numeric vector for history statistics (codes for classification)."""
        try:
            target_values = self.train_df[self.profile.target_column]
            if self.profile.is_regression:
                y_series = pd.to_numeric(target_values, errors="coerce")
                med = y_series.median()
                return y_series.fillna(med if pd.notna(med) else 0.0).values.astype(float)
            codes, _ = pd.factorize(target_values.astype(str), sort=True)
            return codes.astype(float)
        except Exception:
            return None

    def _init_hierarchy(self):
        """Select moderate-cardinality attribute columns and build time-safe history.

        Each candidate column defines a *coarse* aggregation level: the target history of
        its values across periods.  Selection is heuristic — the highest correlation ratio
        (eta squared) with the target among columns with 2..60 distinct values — so the
        block works for regression and classification, with or without a detected entity
        key.  The block stays disabled when no time axis or no usable candidate exists.
        """
        self.hier = []
        self.hier_enabled = False
        self.hier_cols = []
        if self.profile.task_type == "multilabel":
            return
        if self.period_tr is None or len(self.periods) < 2 or not self.cat_features:
            return
        y = self._target_numeric()
        if y is None or len(y) == 0:
            return
        n_tr = len(self.train_df)
        scored = []
        for c in list(self.cat_features):
            try:
                nuniq = int(self.X_tr[c].nunique(dropna=False))
                if not (2 <= nuniq <= 60):
                    continue
                combined = pd.concat([self.X_tr[c].astype(str),
                                      self.X_te[c].astype(str) if len(self.X_te) else
                                      pd.Series([], dtype=str)], ignore_index=True)
                codes_all = pd.factorize(combined, sort=False)[0].astype(np.int64)
                codes_tr = codes_all[:n_tr]
                codes_te = codes_all[n_tr:]
                valid = (codes_tr >= 0) & np.isfinite(y)
                if valid.sum() < max(50, 0.3 * n_tr):
                    continue
                eta = _eta_squared(codes_tr[valid], y[valid])
                if eta is None or not np.isfinite(eta):
                    continue
                scored.append((float(eta), c, codes_tr, codes_te))
            except Exception:
                continue
        if not scored:
            return
        scored.sort(key=lambda t: t[0], reverse=True)
        best_eta = scored[0][0]
        chosen = [t for t in scored[:3] if t[0] >= max(0.004, 0.15 * best_eta)]
        n_entities = int(max(int(t[2].max()) for t in chosen)) + 1 if chosen else 0
        for eta, c, codes_tr, codes_te in chosen:
            try:
                n_ent = int(max(codes_tr.max(), codes_te.max() if len(codes_te) else -1)) + 1
                history = EntityHistory(codes_tr, self.period_tr, y, n_ent,
                                        list(range(len(self.periods))))
                self.hier.append({"col": c, "eta": eta, "history": history,
                                  "codes_tr": codes_tr, "codes_te": codes_te})
                self.hier_cols.append(c)
            except Exception as exc:  # noqa: BLE001
                if self.verbose:
                    print(f"[FeatureArchitect] hier-признаки для {c} недоступны: {exc}")
                continue
        self.hier_enabled = len(self.hier) > 0
        if self.verbose and self.hier:
            desc = ", ".join(f"{h['col']}(η²={h['eta']:.3f})" for h in self.hier)
            print(f"[FeatureArchitect] hier-признаки включены: {desc}")

    def _init_entity(self):
        prof = self.profile
        if prof.task_type == "multilabel":
            return
        if prof.is_classification and self.period_tr is None:
            if self.verbose:
                print("[FeatureArchitect] сущность найдена, но нет временной оси: "
                      "entity-признаки для классификации пропущены.")
            return
        tr_codes, te_codes, n_entities = make_entity_codes(
            self.train_df, self.test_df, prof.entity_columns)
        target_values = self.train_df[prof.target_column]
        if prof.is_regression:
            y_series = pd.to_numeric(target_values, errors="coerce")
            med = y_series.median()
            y = y_series.fillna(med if pd.notna(med) else 0.0).values.astype(float)
        else:
            codes, _ = pd.factorize(target_values.astype(str), sort=True)
            y = codes.astype(float)
        if self.period_tr is None:
            # no temporal axis: folds cannot be simulated -> entity block stays disabled
            return
        if len(self.periods) < 2:
            return
        try:
            self.entity = EntityHistory(tr_codes, self.period_tr, y, n_entities,
                                       list(range(len(self.periods))))
            self.entity_tr_codes, self.entity_te_codes = tr_codes, te_codes
            self.entity_enabled = True
            if self.verbose:
                print(f"[FeatureArchitect] entity-признаки включены: {prof.entity_columns} "
                      f"(сущностей: {n_entities})")
        except Exception as exc:  # noqa: BLE001
            if self.verbose:
                print(f"[FeatureArchitect] entity-признаки недоступны: {exc}")
            self.entity = None
            self.entity_enabled = False

    # ------------------------------------------------------------------ #
    # feature blocks
    # ------------------------------------------------------------------ #
    def add_frequency_features(self) -> bool:
        if "freq" in self.blocks or not self.cat_features:
            return False
        base_cats = [c for c in self.cat_features if not c.startswith("x_")]
        added = 0
        for c in base_cats:
            try:
                combined = pd.concat([self.X_tr[c].astype(str), self.X_te[c].astype(str)],
                                     ignore_index=True)
                counts = combined.value_counts(dropna=False)
                total = max(len(combined), 1)
                freq = {k: float(v) / total for k, v in counts.items()}
                self.X_tr[f"freq_{c}"] = self.X_tr[c].astype(str).map(freq).fillna(0.0).astype(np.float32)
                self.X_te[f"freq_{c}"] = self.X_te[c].astype(str).map(freq).fillna(0.0).astype(np.float32)
                added += 1
            except Exception:
                continue
        if added:
            self.blocks.append("freq")
        return added > 0

    def add_cross_features(self, max_crosses: int = 3) -> bool:
        if "crosses" in self.blocks:
            return False
        candidates = []
        for c in self.cat_features:
            try:
                n = int(self.X_tr[c].nunique(dropna=False))
                if 2 <= n <= 40:
                    candidates.append((n, c))
            except Exception:
                continue
        candidates.sort()
        candidates = [c for _, c in candidates[:6]]
        pairs = []
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                a, b = candidates[i], candidates[j]
                try:
                    card = self.X_tr[a].nunique() * self.X_tr[b].nunique()
                except Exception:
                    continue
                if 4 <= card <= 500:
                    pairs.append((card, a, b))
        pairs.sort()
        added = 0
        for card, a, b in pairs[:max_crosses]:
            try:
                name = f"x_{a}__{b}"
                self.X_tr[name] = (self.X_tr[a].astype(str) + "|" + self.X_tr[b].astype(str))
                self.X_te[name] = (self.X_te[a].astype(str) + "|" + self.X_te[b].astype(str))
                self.cat_features.append(name)
                added += 1
            except Exception:
                continue
        if added:
            self.blocks.append("crosses")
        return added > 0

    def add_entity_block(self) -> bool:
        if not self.entity_enabled or "entity" in self.blocks:
            return False
        self.blocks.append("entity")
        return True

    def add_numeric_interactions(self, max_pairs: int = 10, top_k: int = 6) -> bool:
        """Ratios/products of the most target-correlated numeric columns (generic FE)."""
        if "numx" in self.blocks:
            return False
        y = self._target_numeric()
        if y is None:
            return False
        ranked = []
        y = np.asarray(y, dtype=float)
        for c in self.num_features:
            try:
                v = self.X_tr[c].values.astype(float)
                m = np.isfinite(v) & np.isfinite(y)
                if m.sum() < max(200, 0.05 * len(v)):
                    continue
                nuniq = len(np.unique(v[m]))
                if nuniq < 5:
                    continue
                if self.profile.is_classification:
                    bins = np.unique(np.quantile(v[m], np.linspace(0, 1, 11)))
                    if len(bins) < 3:
                        continue
                    codes = np.digitize(v[m], bins[1:-1])
                    eta = _eta_squared(codes, y[m])
                    score = eta if eta is not None else 0.0
                else:
                    vv = v[m] - v[m].mean()
                    yy = y[m] - y[m].mean()
                    denom = np.sqrt((vv ** 2).sum() * (yy ** 2).sum())
                    score = abs(float((vv * yy).sum() / denom)) if denom > 0 else 0.0
                if np.isfinite(score):
                    ranked.append((score, c))
            except Exception:
                continue
        if len(ranked) < 2:
            return False
        ranked.sort(reverse=True)
        top = [c for sc, c in ranked[:top_k] if sc >= 0.02]
        if len(top) < 2:
            return False
        eps = 1e-6
        added = 0
        for a, b in itertools.islice(itertools.combinations(top, 2), max_pairs):
            try:
                va, vb = self.X_tr[a].values.astype(float), self.X_tr[b].values.astype(float)
                ta, tb = self.X_te[a].values.astype(float), self.X_te[b].values.astype(float)
                rname = f"ratio_{a}_{b}"
                self.X_tr[rname] = np.clip(va / (np.abs(vb) + eps), -1e6, 1e6).astype(np.float32)
                self.X_te[rname] = np.clip(ta / (np.abs(tb) + eps), -1e6, 1e6).astype(np.float32)
                pname = f"prod_{a}_{b}"
                self.X_tr[pname] = np.clip(va * vb, -1e9, 1e9).astype(np.float32)
                self.X_te[pname] = np.clip(ta * tb, -1e9, 1e9).astype(np.float32)
                added += 2
            except Exception:
                continue
        if added:
            self.blocks.append("numx")
        return added > 0

    def add_group_relative_features(self, top_k: int = 4, top_cats: int = 2) -> bool:
        """Label-free group-relative numerics: x - mean_{group}(x).

        Complements raw values with "how unusual is this row inside its own category",
        which is what trees otherwise have to reconstruct with many splits.
        """
        if "relx" in self.blocks:
            return False
        y = self._target_numeric()
        y_arr = np.asarray(y, dtype=float) if y is not None else None

        def _score(col_values, numeric):
            try:
                v = np.asarray(col_values, dtype=float)
                m = np.isfinite(v)
                if m.sum() < max(200, 0.05 * len(v)) or len(np.unique(v[m])) < 5:
                    return 0.0
                if y_arr is None:
                    return 0.0
                if self.profile.is_classification:
                    codes = pd.Series(v).rank(method="dense").astype(int).values
                    eta = _eta_squared(codes, y_arr)
                    return float(eta) if eta is not None and np.isfinite(eta) else 0.0
                vv = v[m] - v[m].mean()
                yy = y_arr[m] - np.nanmean(y_arr[m])
                denom = np.sqrt((vv ** 2).sum() * (yy ** 2).sum())
                return abs(float((vv * yy).sum() / denom)) if denom > 0 else 0.0
            except Exception:
                return 0.0

        cat_cands = []
        for c in self.cat_features:
            try:
                n = int(self.X_tr[c].nunique())
            except Exception:
                continue
            if 2 <= n <= 60:
                sc = _score(pd.Series(self.X_tr[c].astype(str)).astype("category").cat.codes,
                            False)
                cat_cands.append((sc, c))
        num_cands = [(sc, c) for sc, c in ((_score(self.X_tr[c].values, True), c)
                                           for c in self.num_features)]
        if not cat_cands or not num_cands:
            return False
        cat_cands.sort(reverse=True)
        num_cands.sort(reverse=True)
        cats = [c for _, c in cat_cands[:top_cats]]
        nums = [c for _, c in num_cands[:top_k]]
        added = 0
        for cat in cats:
            for num in nums:
                try:
                    gtr = self.X_tr.groupby(self.X_tr[cat].astype(str), observed=True)[num]
                    means = gtr.mean().astype(float)
                    glob = float(np.nanmean(means.values)) if len(means) else 0.0
                    name = f"rel_{cat}_{num}"
                    self.X_tr[name] = (
                        self.X_tr[num].astype(float)
                        - self.X_tr[cat].astype(str).map(means).astype(float)
                    ).astype(np.float32)
                    self.X_te[name] = (
                        self.X_te[num].astype(float)
                        - self.X_te[cat].astype(str).map(means).astype(float)
                    ).astype(np.float32)
                    for frame in (self.X_tr, self.X_te):
                        frame[name] = frame[name].fillna(glob).astype(np.float32)
                    added += 1
                except Exception:
                    continue
        if added:
            self.blocks.append("relx")
        return added > 0

    def add_hierarchy_block(self) -> bool:
        if not self.hier or "hier" in self.blocks:
            return False
        self.hier_enabled = True
        self.blocks.append("hier")
        return True

    def drop_block(self, name: str):
        if name in self.blocks:
            self.blocks.remove(name)
        if name in ("numx", "relx"):
            prefixes = ("ratio_", "prod_") if name == "numx" else ("rel_",)
            for c in [c for c in self.X_tr.columns if c.startswith(prefixes)]:
                self.X_tr.drop(columns=[c], inplace=True, errors="ignore")
                self.X_te.drop(columns=[c], inplace=True, errors="ignore")
        if name == "crosses":
            for c in [c for c in self.cat_features if c.startswith("x_")]:
                self.cat_features.remove(c)

    # ------------------------------------------------------------------ #
    # row materialisation
    # ------------------------------------------------------------------ #
    def entity_features(self, side: str, rows: Optional[np.ndarray], mode: str) -> Optional[pd.DataFrame]:
        if not self.has_entity:
            return None
        if side == "test":
            codes = self.entity_te_codes
            pidx = np.where(self.period_te >= 0, self.period_te, self.test_period_code) \
                if self.period_te is not None else np.zeros(len(self.test_df), dtype=np.int64)
            if rows is not None:
                codes = codes[rows]
                pidx = pidx[rows]
            return self.entity.feature_matrix(codes, pidx, MODE_ALL)
        codes = self.entity_tr_codes
        pidx = self.period_tr
        if rows is not None:
            codes = codes[rows]
            pidx = pidx[rows]
        return self.entity.feature_matrix(codes, pidx, mode)

    def hierarchy_features(self, side: str, rows: Optional[np.ndarray],
                           mode: str) -> Optional[pd.DataFrame]:
        """Time-safe target history of coarse attribute groups (one block per candidate)."""
        if not self.has_hierarchy:
            return None
        use_mode = MODE_ALL if side == "test" else mode
        frames = []
        for item in self.hier:
            try:
                codes = item["codes_te"] if side == "test" else item["codes_tr"]
                pidx = self.period_te if side == "test" else self.period_tr
                if rows is not None:
                    codes = codes[rows]
                    pidx = pidx[rows] if pidx is not None else None
                if pidx is None:
                    pidx = np.zeros(len(codes), dtype=np.int64)
                fm = item["history"].feature_matrix(codes, pidx, use_mode)
                keep = [k for k in ("ent_mean", "ent_count", "ent_std", "ent_last")
                        if k in fm.columns]
                if not keep:
                    continue
                fm = fm[keep]
                fm.columns = [f"h_{item['col']}_{k.replace('ent_', '')}" for k in keep]
                frames.append(fm.reset_index(drop=True))
            except Exception:
                continue
        if not frames:
            return None
        return pd.concat(frames, axis=1)

    def frame(self, side: str, rows: Optional[np.ndarray] = None, mode: str = MODE_LOO,
              include_entity: Optional[bool] = None) -> pd.DataFrame:
        """Materialise the feature matrix for a side ('train'|'test') and row subset."""
        base = self.X_tr if side == "train" else self.X_te
        base = base if rows is None else base.iloc[rows]
        if "crosses" not in self.blocks:
            keep = [c for c in base.columns if not c.startswith("x_")]
            base = base[keep]
        use_entity = self.has_entity if include_entity is None else (include_entity and self.has_entity)
        if use_entity:
            ent = self.entity_features(side, rows, mode)
            if ent is not None:
                base = pd.concat([base.reset_index(drop=True), ent.reset_index(drop=True)], axis=1)
        if self.has_hierarchy:
            hier = self.hierarchy_features(side, rows, mode)
            if hier is not None:
                base = pd.concat([base.reset_index(drop=True), hier.reset_index(drop=True)], axis=1)
        return base

    def cat_features_for(self, frame: pd.DataFrame) -> List[str]:
        out = []
        for c in frame.columns:
            if c not in self.cat_features:
                continue
            dt = frame[c].dtype
            if pd.api.types.is_numeric_dtype(dt) or pd.api.types.is_bool_dtype(dt):
                continue
            out.append(c)
        return out

    def describe(self) -> str:
        hier_txt = f", hier: {self.hier_cols}" if self.hier else ""
        return (f"колонок в базовом наборе: {self.X_tr.shape[1]}, блоки: {self.blocks}, "
                f"категориальных: {len(self.cat_features)}, entity: {self.has_entity}"
                f"{hier_txt}")


# --------------------------------------------------------------------------- #
# LLM-guided feature generation (bounded, optional — never on the critical path)
# --------------------------------------------------------------------------- #
class LLMFeatureGenerator:
    """Asks the platform LLM for a feature formula and executes it in a restricted scope."""

    BANNED = ("import ", "open(", "exec(", "eval(", "subprocess", "os.", "sys.", "__",
              "read_csv", "to_csv", "socket", "requests", "urllib", "shutil", "pathlib")

    def __init__(self, profile: TaskProfile, train_df: pd.DataFrame, task_text: str,
                 verbose: bool = True):
        self.profile = profile
        self.train_df = train_df
        self.task_text = task_text
        self.verbose = verbose
        self.model = os.environ.get("OPENAI_MODEL", "qwen/qwen3.8-27b")
        try:
            from task_profiler import build_llm_client
            self.client = build_llm_client(timeout=75.0)
        except Exception:
            self.client = None

    @property
    def available(self) -> bool:
        return self.client is not None

    def propose(self, history: List[str]) -> Optional[Dict[str, str]]:
        if not self.available:
            return None
        cols = ", ".join(map(str, self.train_df.columns[:60]))
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": FEATURE_SYSTEM_PROMPT},
                          {"role": "user", "content": feature_user_prompt(
                              self.task_text, cols, "\n".join(history[-5:]) or "нет")}],
                response_format={"type": "json_object"},
                temperature=0.4, max_tokens=900,
            )
            content = resp.choices[0].message.content
            if not content:
                return None
            data = json.loads(content)
            code = str(data.get("python_code") or "")
            if not code or "def add_features" not in code:
                return None
            if any(b in code for b in self.BANNED):
                if self.verbose:
                    print("[LLMFeatures] предложение отклонено фильтром безопасности")
                return None
            return {"hypothesis": str(data.get("hypothesis", "llm_feature")), "code": code}
        except Exception as exc:  # noqa: BLE001
            if self.verbose:
                print(f"[LLMFeatures] LLM недоступен ({type(exc).__name__}), пропуск итерации.")
            return None

    def materialise(self, code: str, df: pd.DataFrame, timeout_s: float = 90.0) -> Optional[pd.DataFrame]:
        """Execute generated code on a capped frame, validate and extend to the full frame."""
        namespace: Dict[str, Any] = {"np": np, "pd": pd, "MISSING": MISSING_TOKEN}
        t0 = time.time()
        try:
            exec(compile(code, "<llm_feature>", "exec"), namespace)  # noqa: S102
            fn = namespace.get("add_features")
            if not callable(fn):
                return None
            if len(df) > 30000:
                sample = df.iloc[:30000].copy()
                out = fn(sample)
                if not isinstance(out, pd.DataFrame) or len(out) != len(sample):
                    return None
                new_cols = [c for c in out.columns if c not in sample.columns]
                if not new_cols or len(new_cols) > 8:
                    return None
                out_full = fn(df.copy())
                if not isinstance(out_full, pd.DataFrame) or len(out_full) != len(df):
                    return None
                src = out_full
            else:
                out = fn(df.copy())
                if not isinstance(out, pd.DataFrame) or len(out) != len(df):
                    return None
                src = out
            new_cols = [c for c in src.columns if c not in df.columns]
            if not new_cols or len(new_cols) > 8 or time.time() - t0 > timeout_s:
                return None
            result = pd.DataFrame(index=df.index)
            for c in new_cols:
                vals = pd.to_numeric(pd.Series(src[c].values), errors="coerce")
                result[c] = vals.astype(np.float32)
            return result
        except Exception as exc:  # noqa: BLE001
            if self.verbose:
                print(f"[LLMFeatures] исполнение кода не удалось: {exc}")
            return None
