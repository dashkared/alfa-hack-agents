"""Agent 3 — ModelPortfolioAgent.

Owns the validation scheme, the heterogeneous model portfolio, constrained blending and
the decision-rule (class multiplier) calibration for Macro-F1-like metrics.

Hard rules baked in after the platform audit:
  * probabilities are always scattered into a full-width matrix (`classes_` mapping) so a
    fold whose training part misses a rare class never crashes and never shifts columns;
  * every fit path is wrapped in try/except — one broken model cannot stop the pipeline;
  * every loop honours a deadline callback and degrades gracefully.
"""
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from entity_features import MODE_ALL, MODE_BEFORE, MODE_LOO
from evaluation import compute_metric, is_better, metric_direction
from prompts import TaskProfile

warnings.filterwarnings("ignore")
SEED = 42


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _to_category(X: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
    X = X.copy()
    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].astype("category")
    return X


def _align_categories(X_tr, X_va, X_te, cat_cols: List[str]):
    """Category dtype with ONE shared dictionary across train/valid/test.

    Casting each frame separately makes pandas assign codes from that frame's own sorted
    values, so a category missing here and there silently shifts the integer codes that
    LightGBM consumes — i.e. validation sees different numbers than training did.
    """
    frames = [X_tr.copy(), X_va.copy(), X_te.copy()]
    for c in cat_cols:
        try:
            if not any(c in f.columns for f in frames):
                continue
            values = pd.concat([f[c].astype(str) for f in frames if c in f.columns],
                               ignore_index=True)
            cats = sorted(pd.unique(values).tolist())
            for f in frames:
                if c in f.columns:
                    f[c] = pd.Categorical(f[c].astype(str), categories=cats)
        except Exception:
            continue
    return frames[0], frames[1], frames[2]


def _ordinal_encode(X_tr: pd.DataFrame, X_va: pd.DataFrame, cat_cols: List[str],
                    X_te: Optional[pd.DataFrame] = None):
    outs_tr, outs_va = X_tr.copy(), X_va.copy()
    outs_te = X_te.copy() if X_te is not None else None
    for c in cat_cols:
        if c not in X_tr.columns:
            continue
        try:
            mapping = {v: i for i, v in enumerate(pd.unique(X_tr[c].astype(str)))}
            outs_tr[c] = X_tr[c].astype(str).map(mapping).fillna(-1).astype(np.int32)
            outs_va[c] = X_va[c].astype(str).map(mapping).fillna(-1).astype(np.int32)
            if outs_te is not None:
                outs_te[c] = X_te[c].astype(str).map(mapping).fillna(-1).astype(np.int32)
        except Exception:
            continue
    return outs_tr, outs_va, outs_te


def _scatter_proba(proba: np.ndarray, classes, n_classes: int) -> np.ndarray:
    proba = np.asarray(proba, dtype=float)
    if proba.ndim == 1:
        proba = proba.reshape(-1, 1)
    if proba.shape[1] == n_classes:
        return proba
    full = np.zeros((proba.shape[0], n_classes), dtype=float)
    for j, cls in enumerate(np.asarray(classes).astype(int)):
        if 0 <= cls < n_classes:
            full[:, cls] = proba[:, j]
    return full


class ModelPortfolioAgent:
    def __init__(self, profile: TaskProfile, n_jobs: int = 4, seed: int = SEED,
                 verbose: bool = True, deadline: Optional[Callable[[], float]] = None):
        self.profile = profile
        self.n_jobs = max(1, int(n_jobs))
        self.seed = seed
        self.verbose = verbose
        self.deadline = deadline or (lambda: 1e9)
        self.results: Dict[str, Dict[str, Any]] = {}
        self.last_fold_scores: List[float] = []
        self.blend_info: Dict[str, Any] = {}
        self._y_ref: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    def remaining(self) -> float:
        try:
            return float(self.deadline())
        except Exception:
            return 1e9

    def log(self, msg: str):
        if self.verbose:
            print(msg, flush=True)

    def model_families(self) -> List[str]:
        """Heterogeneous portfolio, ordered by expected strength on tabular data."""
        if self.profile.is_classification:
            return ["lgbm", "catboost", "xgboost", "histgb", "lgbm:deep", "lgbm:ovr",
                    "linear"]
        return ["catboost", "lgbm", "xgboost", "histgb", "linear", "catboost:deep",
                "catboost:resid", "lgbm:resid"]

    def score(self, y_true: np.ndarray, pred=None, proba: Optional[np.ndarray] = None) -> float:
        metric = self.profile.metric
        if self.profile.is_classification:
            if proba is not None and str(metric).lower() in ("roc-auc", "auc", "logloss", "log loss"):
                return compute_metric(y_true, None, metric, proba=proba)
            if pred is None and proba is not None:
                pred = np.argmax(proba, axis=1)
            return compute_metric(y_true, pred, metric, proba=proba)
        return compute_metric(y_true, pred, metric)

    @property
    def direction(self) -> str:
        return metric_direction(self.profile.metric)

    def better(self, a: float, b: Optional[float]) -> bool:
        return is_better(a, b, self.direction, eps=1e-9)

    # ------------------------------------------------------------------ #
    # validation scheme
    # ------------------------------------------------------------------ #
    def make_folds(self, y: np.ndarray, bundle, n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Expanding-window folds for panel/future-period data, otherwise K-fold."""
        n = len(y)
        periods = getattr(bundle, "period_tr", None)
        n_periods = len(getattr(bundle, "periods", []) or [])
        if periods is not None and n_periods >= 3:
            uniq = sorted(pd.unique(periods))
            # the test split is a *future* period: a validation fold is only representative
            # when the model has seen a comparable amount of history.  With 5+ periods we
            # therefore start validating after three training periods (a fold trained on two
            # periods is far from the inference setting and only distorts model selection).
            min_train_periods = 3 if n_periods >= 5 else (2 if n_periods >= 4 else 1)
            folds = []
            for idx, p in enumerate(uniq):
                if idx < min_train_periods:
                    continue
                tr = np.where(np.isin(periods, uniq[:idx]))[0]
                va = np.where(periods == p)[0]
                if len(tr) < 100 or len(va) < 20:
                    continue
                folds.append((tr, va))
            if folds:
                keep = min(len(folds), 3)
                return folds[-keep:]
        n_splits = int(min(n_splits, max(2, n // 40)))
        if self.profile.is_classification:
            try:
                counts = np.bincount(np.asarray(y).astype(int))
                counts = counts[counts > 0]
                if len(counts):
                    n_splits = int(max(2, min(n_splits, int(counts.min()))))
            except Exception:
                pass
        from sklearn.model_selection import KFold, StratifiedKFold
        if self.profile.is_classification:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
            return list(skf.split(np.zeros(n), np.asarray(y).astype(int)))
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        return list(kf.split(np.zeros(n)))

    # ------------------------------------------------------------------ #
    # single model fit
    # ------------------------------------------------------------------ #
    def _fit_predict(self, family: str, X_tr: pd.DataFrame, y_tr: np.ndarray,
                     X_va: pd.DataFrame, y_va: np.ndarray, X_te: pd.DataFrame,
                     cat_cols: List[str], n_classes: int, fast: bool = False,
                     seed: Optional[int] = None) -> Optional[Dict[str, Any]]:
        is_cls = self.profile.is_classification
        seed = self.seed if seed is None else seed
        nj = self.n_jobs
        iters = 400 if fast else 2500
        early = 50 if fast else 120
        if is_cls and n_classes > 4 and not fast:
            # multiclass CatBoost/XGBoost grow one tree per class per iteration: cap them so
            # the whole portfolio still fits the budget (LightGBM carries the accuracy here)
            iters, early = 900, 70
        family, _, tag = str(family).partition(":")
        try:
            if family == "catboost":
                from catboost import CatBoostClassifier, CatBoostRegressor
                cats = [c for c in cat_cols if c in X_tr.columns]
                common = dict(iterations=iters,
                              learning_rate=0.12 if fast else 0.06,
                              random_seed=seed, verbose=0, thread_count=nj,
                              od_type="Iter", od_wait=early,
                              allow_writing_files=False)
                depth = 6 if is_cls else 7
                if tag == "deep" and not fast:
                    common.update(dict(iterations=6000, learning_rate=0.03, od_wait=250))
                    depth = 8
                if is_cls:
                    model = CatBoostClassifier(
                        loss_function="MultiClass" if n_classes > 2 else "Logloss",
                        depth=depth, **common)
                else:
                    model = CatBoostRegressor(loss_function="RMSE", depth=depth, **common)
                model.fit(X_tr, y_tr, cat_features=cats, eval_set=(X_va, y_va))
                best_iter = int(model.get_best_iteration() or 0) + 1
                return self._pack(model, X_va, X_te, n_classes, is_cls, best_iter)

            if family == "lgbm":
                import lightgbm as lgb
                Xt, Xv, Xe = _align_categories(X_tr, X_va, X_te, cat_cols)
                params = dict(n_estimators=iters,
                              learning_rate=0.12 if fast else 0.05,
                              num_leaves=63 if fast else 127,
                              min_child_samples=20, subsample=0.85, subsample_freq=1,
                              colsample_bytree=0.7, reg_lambda=1.0, n_jobs=nj,
                              random_state=seed, verbose=-1)
                if tag == "deep" and not fast:
                    params.update(dict(n_estimators=5000, learning_rate=0.03,
                                       num_leaves=255, min_child_samples=40,
                                       colsample_bytree=0.5))
                if is_cls:
                    model = lgb.LGBMClassifier(
                        objective="multiclass" if n_classes > 2 else "binary",
                        num_class=n_classes if n_classes > 2 else 1, **params)
                else:
                    model = lgb.LGBMRegressor(objective="regression", **params)
                model.fit(Xt, y_tr, eval_set=[(Xv, y_va)],
                          callbacks=[lgb.early_stopping(early, verbose=False),
                                     lgb.log_evaluation(0)])
                best_iter = int(getattr(model, "best_iteration_", 0) or iters)
                return self._pack(model, Xv, Xe, n_classes, is_cls, best_iter)

            if family == "xgboost":
                import xgboost as xgb
                Xt, Xv, Xe = _ordinal_encode(X_tr, X_va, cat_cols, X_te)
                params = dict(n_estimators=iters, learning_rate=0.12 if fast else 0.06,
                              max_depth=7, min_child_weight=1.0, subsample=0.85,
                              colsample_bytree=0.7, reg_lambda=1.0, n_jobs=nj,
                              random_state=seed, tree_method="hist",
                              early_stopping_rounds=early, verbosity=0)
                if tag == "deep" and not fast:
                    params.update(dict(n_estimators=5000, learning_rate=0.03,
                                       max_depth=9, min_child_weight=3.0,
                                       colsample_bytree=0.5))
                if is_cls:
                    model = xgb.XGBClassifier(
                        objective="multi:softprob" if n_classes > 2 else "binary:logistic",
                        num_class=n_classes if n_classes > 2 else None, **params)
                else:
                    model = xgb.XGBRegressor(objective="reg:squarederror", **params)
                model.fit(Xt, y_tr, eval_set=[(Xv, y_va)], verbose=False)
                best_iter = int(getattr(model, "best_iteration", 0) or iters)
                return self._pack(model, Xv, Xe, n_classes, is_cls, best_iter)

            if family == "histgb":
                from sklearn.ensemble import (HistGradientBoostingClassifier,
                                              HistGradientBoostingRegressor)
                Xt, Xv, Xe = _ordinal_encode(X_tr, X_va, cat_cols, X_te)
                common = dict(max_iter=140 if fast else 600,
                              learning_rate=0.15 if fast else 0.07,
                              max_leaf_nodes=63, l2_regularization=1.0,
                              early_stopping=True, validation_fraction=0.1,
                              n_iter_no_change=30, random_state=seed)
                model = HistGradientBoostingClassifier(**common) if is_cls \
                    else HistGradientBoostingRegressor(loss="squared_error", **common)
                model.fit(Xt, y_tr)
                return self._pack(model, Xv, Xe, n_classes, is_cls, None)

            if family == "lgbm" and tag == "ovr" and is_cls:
                import lightgbm as _lgb
                params = dict(n_estimators=iters,
                              learning_rate=0.12 if fast else 0.06,
                              num_leaves=63 if fast else 96,
                              min_child_samples=20, subsample=0.85, subsample_freq=1,
                              colsample_bytree=0.6, reg_lambda=1.0, n_jobs=nj,
                              random_state=seed, verbose=-1)
                Xt, Xv, Xe = _align_categories(X_tr, X_va, X_te, cat_cols)
                y_arr = np.asarray(y_tr).astype(int)
                va_arr = np.asarray(y_va).astype(int)
                P_v = np.zeros((len(X_va), n_classes))
                P_e = np.zeros((len(X_te), n_classes))
                for c in range(n_classes):
                    yb = (y_arr == c).astype(int)
                    if yb.sum() < 2 or yb.sum() == len(yb):
                        P_v[:, c] = float(yb.mean())
                        P_e[:, c] = float(yb.mean())
                        continue
                    model = _lgb.LGBMClassifier(objective="binary", **params)
                    vb = (va_arr == c).astype(int)
                    if len(np.unique(vb)) > 1:
                        model.fit(Xt, yb, eval_set=[(Xv, vb)],
                                  callbacks=[_lgb.early_stopping(early, verbose=False)],
                                  categorical_feature=cat_cols or "auto")
                    else:
                        model.fit(Xt, yb, categorical_feature=cat_cols or "auto")
                    P_v[:, c] = model.predict_proba(Xv)[:, 1]
                    P_e[:, c] = model.predict_proba(Xe)[:, 1]
                P_v = P_v / np.clip(P_v.sum(axis=1, keepdims=True), 1e-12, None)
                P_e = P_e / np.clip(P_e.sum(axis=1, keepdims=True), 1e-12, None)
                return {"proba": P_v, "test_proba": P_e, "pred": P_v.argmax(axis=1),
                        "test_pred": P_e.argmax(axis=1), "best_iter": None}

            if family == "linear":
                from sklearn.linear_model import LogisticRegression, Ridge
                from sklearn.pipeline import Pipeline
                from sklearn.preprocessing import StandardScaler
                Xt, Xv, Xe = _ordinal_encode(X_tr, X_va, cat_cols, X_te)
                med = Xt.median(numeric_only=True)
                Xt = Xt.fillna(med).fillna(0.0)
                Xv = Xv.fillna(med).fillna(0.0)
                Xe = Xe.fillna(med).fillna(0.0)
                if is_cls:
                    model = Pipeline([("sc", StandardScaler(with_mean=False)),
                                      ("clf", LogisticRegression(
                                          max_iter=600, C=1.0, n_jobs=nj,
                                          random_state=seed))])
                else:
                    model = Pipeline([("sc", StandardScaler()),
                                      ("ridge", Ridge(alpha=2.0, random_state=seed))])
                model.fit(Xt, y_tr)
                return self._pack(model, Xv, Xe, n_classes, is_cls, None)

            if family == "extra_trees":
                from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
                Xt, Xv, Xe = _ordinal_encode(X_tr, X_va, cat_cols, X_te)
                Xt, Xv, Xe = Xt.fillna(-999), Xv.fillna(-999), Xe.fillna(-999)
                if is_cls:
                    model = ExtraTreesClassifier(n_estimators=350, max_features="sqrt",
                                                 n_jobs=nj, random_state=seed,
                                                 min_samples_leaf=2)
                else:
                    model = ExtraTreesRegressor(n_estimators=300, max_features=0.6,
                                                n_jobs=nj, random_state=seed,
                                                min_samples_leaf=2)
                model.fit(Xt, y_tr)
                return self._pack(model, Xv, Xe, n_classes, is_cls, None)
        except Exception as exc:  # noqa: BLE001
            self.log(f"      ⚠️ {family}: ошибка обучения ({type(exc).__name__}: {exc})")
            return None
        return None

    def _pack(self, model, X_va, X_te, n_classes: int, is_cls: bool,
              best_iter: Optional[int]) -> Dict[str, Any]:
        if is_cls:
            classes = np.asarray(getattr(model, "classes_", np.arange(n_classes)))
            pv = _scatter_proba(model.predict_proba(X_va), classes, n_classes)
            pt = _scatter_proba(model.predict_proba(X_te), classes, n_classes)
            pv = pv / np.clip(pv.sum(axis=1, keepdims=True), 1e-12, None)
            pt = pt / np.clip(pt.sum(axis=1, keepdims=True), 1e-12, None)
            return {"proba": pv, "test_proba": pt, "pred": pv.argmax(axis=1),
                    "test_pred": pt.argmax(axis=1), "best_iter": best_iter}
        pv = np.asarray(model.predict(X_va), dtype=float).ravel()
        pt = np.asarray(model.predict(X_te), dtype=float).ravel()
        return {"pred": pv, "test_pred": pt, "best_iter": best_iter}

    # ------------------------------------------------------------------ #
    # cross-validation over the portfolio
    # ------------------------------------------------------------------ #
    def _fold_frames(self, bundle, tr_idx, va_idx, entity_mode):
        X_tr = bundle.frame("train", tr_idx, mode=entity_mode)
        X_va = bundle.frame("train", va_idx, mode=MODE_BEFORE)
        X_te = bundle.frame("test", None, mode=MODE_ALL)
        return X_tr, X_va, X_te

    def run_cv(self, family: str, bundle, y: np.ndarray,
               folds: List[Tuple[np.ndarray, np.ndarray]], fast: bool = False,
               entity_mode: str = MODE_LOO, residual: bool = False,
               seed: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Cross-validate one family (optionally as a residual-on-entity-history model)."""
        is_cls = self.profile.is_classification
        n_classes = int(self.profile.n_classes) if is_cls else 0
        _, _, fam_tag = str(family).partition(":")
        if fam_tag == "resid" and not is_cls:
            residual = True
        n = len(y)
        oof = np.full(n, np.nan, dtype=float)
        oof_proba = np.zeros((n, max(n_classes, 1)), dtype=float) if is_cls else None
        seen = np.zeros(n, dtype=bool)
        test_acc, fold_scores, iters = [], [], []
        for k, (tr_idx, va_idx) in enumerate(folds):
            if self.remaining() <= 0:
                break
            t0 = time.time()
            X_tr, X_va, X_te = self._fold_frames(bundle, tr_idx, va_idx, entity_mode)
            y_tr, y_va = y[tr_idx], y[va_idx]
            cats = bundle.cat_features_for(X_tr)
            if residual and not is_cls and "ent_mean" in X_tr.columns:
                out = self._fit_residual(family, X_tr, y_tr, X_va, y_va, X_te, cats, fast, seed)
            else:
                out = self._fit_predict(family, X_tr, y_tr, X_va, y_va, X_te, cats,
                                        n_classes, fast=fast, seed=seed)
            if out is None:
                continue
            if is_cls:
                oof_proba[va_idx] = out["proba"]
                seen[va_idx] = True
                fold_scores.append(self.score(y_va, out["pred"], out["proba"]))
            else:
                oof[va_idx] = out["pred"]
                fold_scores.append(self.score(y_va, out["pred"]))
            test_acc.append(out["test_proba"] if is_cls else out["test_pred"])
            iters.append(out.get("best_iter") or 0)
            self.log(f"      · {family}{' (residual)' if residual else ''} "
                     f"fold {k + 1}/{len(folds)}: {self.profile.metric}="
                     f"{fold_scores[-1]:.4f} ({time.time() - t0:.0f}s)")
        if not fold_scores:
            return None
        result: Dict[str, Any] = {
            "family": family, "residual": residual, "fold_scores": fold_scores,
            "mean_fold": float(np.mean(fold_scores)),
            "best_iter": int(np.median([i for i in iters if i] or [0])),
            "fast": fast, "entity_mode": entity_mode,
        }
        if is_cls:
            pred = np.argmax(oof_proba, axis=1)
            result.update({"oof_proba": oof_proba, "oof_pred": pred, "oof": pred,
                           "mask": seen,
                           "oof_score": self.score(y[seen], pred[seen], oof_proba[seen]),
                           "test_proba": np.mean(np.stack(test_acc, axis=0), axis=0) if test_acc else None})
            result["test_pred"] = (np.argmax(result["test_proba"], axis=1)
                                   if result["test_proba"] is not None else None)
        else:
            valid = np.isfinite(oof)
            result.update({"oof": oof, "mask": valid,
                           "oof_score": self.score(y[valid], oof[valid]),
                           "test_pred": np.mean(np.stack(test_acc, axis=0), axis=0) if test_acc else None})
        return result

    def _fit_residual(self, family, X_tr, y_tr, X_va, y_va, X_te, cats, fast, seed):
        """Two-part model for panel data (verified on the public benchmark).

        * part B (rows *with* entity history): residual model on all features except the
          history mean itself — raw residuals beat log-space here (35.05 vs 37.01 OOF);
        * part A (rows *without* history): attribute-only model with every entity-history
          column dropped (they are all missing for those rows by construction).
        """
        base_tr = X_tr["ent_mean"].values.astype(float)
        base_va = X_va["ent_mean"].values.astype(float)
        base_te = X_te["ent_mean"].values.astype(float)
        has_tr, has_va, has_te = np.isfinite(base_tr), np.isfinite(base_va), np.isfinite(base_te)
        if has_tr.sum() < 50:
            return None
        ent_all = [c for c in X_tr.columns if c.startswith("ent_")]
        drop_b = [c for c in ("ent_mean",) if c in X_tr.columns]
        feat_b_tr = X_tr.drop(columns=drop_b)
        feat_b_va = X_va.drop(columns=[c for c in drop_b if c in X_va.columns])
        feat_b_te = X_te.drop(columns=[c for c in drop_b if c in X_te.columns])
        drop_a = [c for c in ent_all if c in X_tr.columns]
        feat_a_tr = X_tr.drop(columns=drop_a)
        feat_a_va = X_va.drop(columns=[c for c in drop_a if c in X_va.columns])
        feat_a_te = X_te.drop(columns=[c for c in drop_a if c in X_te.columns])

        res_tr = y_tr[has_tr] - base_tr[has_tr]
        res_va = y_va[has_va] - base_va[has_va]
        out_b = self._fit_predict(family, feat_b_tr[has_tr], res_tr,
                                  feat_b_va[has_va], res_va,
                                  feat_b_te[has_te], cats, 0, fast=fast, seed=seed)
        if out_b is None:
            return None
        pred = np.full(len(X_va), np.nan)
        pred[has_va] = base_va[has_va] + out_b["pred"]
        test_pred = np.full(len(X_te), np.nan)
        test_pred[has_te] = base_te[has_te] + out_b["test_pred"]
        if (~has_tr).sum() >= 50 and (~has_va).any():
            out_a = self._fit_predict(family, feat_a_tr[~has_tr], y_tr[~has_tr],
                                      feat_a_va[~has_va], y_va[~has_va],
                                      feat_a_te[~has_te], cats, 0, fast=fast, seed=seed)
            if out_a is not None:
                pred[~has_va] = out_a["pred"]
                test_pred[~has_te] = out_a["test_pred"]
        fill = float(np.nanmedian(y_tr))
        pred = np.where(np.isfinite(pred), pred, fill)
        test_pred = np.where(np.isfinite(test_pred), test_pred, fill)
        return {"pred": pred, "test_pred": test_pred,
                "best_iter": out_b.get("best_iter"), "proba": None, "test_proba": None}

    def merge_results(self, base: Dict[str, Any], extra: Dict[str, Any],
                      tag: str = "", w_base: float = 0.5) -> Optional[Dict[str, Any]]:
        """Average two CV results of the same family (multi-seed bagging)."""
        try:
            is_cls = self.profile.is_classification
            mask = np.asarray(base["mask"], dtype=bool) & np.asarray(extra["mask"], dtype=bool)
            if mask.sum() < 20:
                return base
            out = dict(base)
            out["family"] = base["family"] + (f"+{tag}" if tag else "")
            out["fold_scores"] = list(np.mean(np.stack([base["fold_scores"],
                                                        extra["fold_scores"]], axis=0),
                                              axis=0))
            out["mean_fold"] = float(np.mean(out["fold_scores"]))
            out["bagged"] = True
            w_base = float(min(1.0, max(0.0, w_base)))
            if is_cls:
                oof = (w_base * np.asarray(base["oof_proba"], dtype=float)
                       + (1.0 - w_base) * np.asarray(extra["oof_proba"], dtype=float))
                oof = oof / np.clip(oof.sum(axis=1, keepdims=True), 1e-12, None)
                out["oof_proba"] = oof
                out["oof_pred"] = np.argmax(oof, axis=1)
                out["oof"] = out["oof_pred"]
                out["mask"] = mask
                out["oof_score"] = self.score(self._y_ref[mask], out["oof_pred"][mask],
                                              oof[mask])
                if base.get("test_proba") is not None and extra.get("test_proba") is not None:
                    tp = (w_base * np.asarray(base["test_proba"], dtype=float)
                          + (1.0 - w_base) * np.asarray(extra["test_proba"], dtype=float))
                    out["test_proba"] = tp / np.clip(tp.sum(axis=1, keepdims=True), 1e-12, None)
                    out["test_pred"] = out["test_proba"].argmax(axis=1)
            else:
                oof = np.where(np.isfinite(base["oof"]), np.asarray(base["oof"], dtype=float),
                               np.asarray(extra["oof"], dtype=float))
                extra_oof = np.asarray(extra["oof"], dtype=float)
                both = np.isfinite(base["oof"]) & np.isfinite(extra_oof)
                oof = np.where(both, w_base * np.asarray(base["oof"], dtype=float)
                               + (1.0 - w_base) * extra_oof, oof)
                out["oof"] = oof
                out["mask"] = mask
                out["oof_score"] = self.score(self._y_ref[mask], oof[mask])
                if base.get("test_pred") is not None and extra.get("test_pred") is not None:
                    out["test_pred"] = (w_base * np.asarray(base["test_pred"], dtype=float)
                                        + (1.0 - w_base) * np.asarray(extra["test_pred"], dtype=float))
            return out
        except Exception as exc:  # noqa: BLE001
            self.log(f"      ⚠️ слияние seed-результатов не удалось ({exc})")
            return base

    # ------------------------------------------------------------------ #
    # blending
    # ------------------------------------------------------------------ #
    def _blend_weights(self, stacked: np.ndarray, y: np.ndarray, is_cls: bool) -> np.ndarray:
        """Non-negative weights summing to 1 (stacked shape: (k, n[, classes]))."""
        k = stacked.shape[0]
        if k == 1:
            return np.ones(1)
        try:
            from scipy.optimize import minimize
            def objective(w):
                w = np.clip(np.asarray(w), 1e-6, None)
                w = w / w.sum()
                combo = np.tensordot(w, stacked, axes=(0, 0))
                if is_cls:
                    return -_loglik(combo, y)
                return compute_metric(y, combo, "rmse")
            res = minimize(objective, np.ones(k) / k, method="SLSQP",
                           bounds=[(0.0, 1.0)] * k,
                           constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}],
                           options={"maxiter": 120, "ftol": 1e-7})
            w = np.clip(res.x, 0, None)
            return w / w.sum() if w.sum() > 0 else np.ones(k) / k
        except Exception:
            return np.ones(k) / k

    def _better_score(self, a: Optional[float], b: Optional[float]) -> bool:
        if a is None or not np.isfinite(a):
            return False
        if b is None or not np.isfinite(b):
            return True
        return (a > b) if self.direction == "maximize" else (a < b)

    def blend(self, y: np.ndarray) -> Optional[Dict[str, Any]]:
        usable = {k: v for k, v in self.results.items() if v is not None}
        if not usable:
            return None
        is_cls = self.profile.is_classification
        order = sorted(usable.items(), key=lambda kv: kv[1]["oof_score"],
                       reverse=(self.direction == "maximize"))
        best_name, best_res = order[0]
        best_score = best_res["oof_score"]
        keep = []
        for name, r in order:
            rel = abs(r["oof_score"] - best_score) / (abs(best_score) + 1e-9)
            if len(keep) == 0 or rel <= 0.20:
                keep.append((name, r))
        keep = keep[:6]
        mask = np.ones(len(y), dtype=bool)
        for _, r in keep:
            mask &= np.asarray(r["mask"])
        if mask.sum() < max(50, 0.05 * len(y)):
            mask = np.ones(len(y), dtype=bool)
        y_mask = y[mask]
        candidates: Dict[str, Dict[str, Any]] = {}
        single_oof = (np.asarray(best_res["oof_proba"], dtype=float) if is_cls
                      else np.asarray(best_res["oof"], dtype=float))
        candidates["single"] = {"score": best_res["oof_score"], "oof": single_oof,
                                "test": best_res["test_proba"] if is_cls else best_res["test_pred"],
                                "info": {"kind": "single", "models": [best_name]}}
        if len(keep) >= 2:
            if is_cls:
                stacked = np.stack([r["oof_proba"][mask] for _, r in keep], axis=0)   # (k,n,cls)
                mean_combo = stacked.mean(axis=0)
                weights = self._blend_weights(stacked, y_mask, True)
                weighted_combo = np.tensordot(weights, stacked, axes=(0, 0))
                mean_score = self.score(y_mask, None, mean_combo)
                weighted_score = self.score(y_mask, None, weighted_combo)
                nested = self._nested_metric(stacked, y_mask, True)
                honest_weighted = nested if nested is not None else weighted_score
                test_stacked = np.stack([r["test_proba"] for _, r in keep], axis=0)
                candidates["mean"] = {"score": mean_score, "oof": mean_combo,
                                      "test": test_stacked.mean(axis=0),
                                      "info": {"kind": "mean", "models": [n for n, _ in keep]}}
                candidates["weighted"] = {
                    "score": honest_weighted, "oof": weighted_combo,
                    "test": np.tensordot(weights, test_stacked, axes=(0, 0)),
                    "info": {"kind": "weighted",
                             "weights": {n: float(w) for (n, _), w in zip(keep, weights)},
                             "nested": honest_weighted}}
            else:
                stacked = np.stack([r["oof"][mask] for _, r in keep], axis=0)          # (k,n)
                mean_combo = stacked.mean(axis=0)
                weights = self._blend_weights(stacked, y_mask, False)
                weighted_combo = np.tensordot(weights, stacked, axes=(0, 0))
                mean_score = self.score(y_mask, mean_combo)
                weighted_score = self.score(y_mask, weighted_combo)
                nested = self._nested_metric(stacked, y_mask, False)
                honest_weighted = nested if nested is not None else weighted_score
                test_stacked = np.stack([r["test_pred"] for _, r in keep], axis=0)
                candidates["mean"] = {"score": mean_score, "oof": mean_combo,
                                      "test": test_stacked.mean(axis=0),
                                      "info": {"kind": "mean", "models": [n for n, _ in keep]}}
                candidates["weighted"] = {
                    "score": honest_weighted, "oof": weighted_combo,
                    "test": np.tensordot(weights, test_stacked, axes=(0, 0)),
                    "info": {"kind": "weighted",
                             "weights": {n: float(w) for (n, _), w in zip(keep, weights)},
                             "nested": honest_weighted}}
        choice, choice_score = "single", None
        for name, cand in candidates.items():
            if self._better_score(cand["score"], choice_score):
                choice, choice_score = name, cand["score"]
        chosen = candidates[choice]
        self.blend_info = chosen["info"]
        return {"oof": chosen["oof"], "test": chosen["test"], "info": chosen["info"],
                "score": float(choice_score), "mask": mask,
                "candidates": {k: float(v["score"]) for k, v in candidates.items()}}

    def _nested_metric(self, stacked: np.ndarray, y: np.ndarray, is_cls: bool) -> Optional[float]:
        n = len(y)
        if n < 200 or stacked.shape[0] < 2:
            return None
        parts = np.array_split(np.arange(n), 4)
        scores = []
        for i, hold in enumerate(parts):
            fit_idx = np.concatenate([p for j, p in enumerate(parts) if j != i])
            if len(fit_idx) < 100 or len(hold) < 20:
                continue
            try:
                w = self._blend_weights(stacked[:, fit_idx], y[fit_idx], is_cls)
                combo = np.tensordot(w, stacked[:, hold], axes=(0, 0))
                scores.append(self.score(y[hold], None, combo) if is_cls
                              else self.score(y[hold], combo))
            except Exception:
                continue
        return float(np.mean(scores)) if scores else None

    # ------------------------------------------------------------------ #
    # classification decision-rule calibration
    # ------------------------------------------------------------------ #
    def tune_class_multipliers(self, proba: np.ndarray, y: np.ndarray,
                               max_rows: int = 30000) -> np.ndarray:
        """Coordinate ascent on per-class probability multipliers (Macro-F1 friendly)."""
        k = proba.shape[1]
        ones = np.ones(k, dtype=float)
        if not self.profile.is_classification:
            return ones
        if str(self.profile.metric).lower() not in ("macro f1", "f1", "weighted f1", "accuracy",
                                                    "balanced accuracy"):
            return ones
        try:
            if len(y) > max_rows:
                rng = np.random.RandomState(self.seed)
                idx = rng.choice(len(y), max_rows, replace=False)
                proba_s, y_s = proba[idx], y[idx]
            else:
                proba_s, y_s = proba, y
            base_score = self.score(y_s, np.argmax(proba_s, axis=1))
            weights = self._bagged_multipliers(proba_s, y_s)
            tuned_score = self.score(y_s, np.argmax(proba_s * weights, axis=1))
            if not self.better(tuned_score, base_score):
                return ones
            replicated = self._nested_multiplier_gain(proba_s, y_s)
            if replicated is not None and replicated <= 0:
                self.log("      · множители классов отклонены (эффект не воспроизводится)")
                return ones
            self.log(f"      · множители классов: {self.profile.metric} {base_score:.4f} -> "
                     f"{tuned_score:.4f} (nested Δ={replicated if replicated is None else round(replicated, 4)})")
            return weights
        except Exception:
            return ones

    def _bagged_multipliers(self, proba: np.ndarray, y: np.ndarray,
                            n_bags: int = 10, shrink: float = 0.7) -> np.ndarray:
        """Bag the coordinate-ascent tuning over row subsamples.

        A single tuning pass flips classes on a handful of rows (macro F1 is dominated by
        rare classes), so the tuned weights are averaged in log space over bootstrap
        resamples and shrunk towards 1.0 — same expected gain, far lower variance.
        """
        k = proba.shape[1]
        logs = []
        n = len(y)
        rng = np.random.RandomState(self.seed)
        for b in range(n_bags):
            if b == 0:
                idx = np.arange(n)
            else:
                idx = rng.choice(n, max(300, int(0.5 * n)), replace=True)
            try:
                w = self.tune_class_multipliers_raw(proba[idx], y[idx])
                logs.append(np.log(np.clip(w, 0.25, 4.0)))
            except Exception:
                continue
        if not logs:
            return np.ones(k, dtype=float)
        mean_log = np.mean(np.stack(logs, axis=0), axis=0)
        w = np.exp(shrink * mean_log)
        return np.clip(w / np.mean(w), 0.35, 3.0)

    def tune_class_multipliers_raw(self, proba: np.ndarray, y: np.ndarray) -> np.ndarray:
        k = proba.shape[1]
        weights = np.ones(k, dtype=float)
        grid = [0.55, 0.7, 0.85, 1.0, 1.2, 1.45, 1.8, 2.3]
        best_score = self.score(y, np.argmax(proba, axis=1))
        for _ in range(2):
            improved = False
            for c in range(k):
                best_w = weights[c]
                for g in grid:
                    trial = weights.copy()
                    trial[c] = g
                    sc = self.score(y, np.argmax(proba * trial, axis=1))
                    if self.better(sc, best_score):
                        best_score, best_w, improved = sc, g, True
                weights[c] = best_w
            if not improved:
                break
        return weights

    def _nested_multiplier_gain(self, proba: np.ndarray, y: np.ndarray) -> Optional[float]:
        n = len(y)
        if n < 300:
            return None
        parts = np.array_split(np.arange(n), 3)
        gains = []
        ones = np.ones(proba.shape[1])
        for i, hold in enumerate(parts):
            fit_idx = np.concatenate([p for j, p in enumerate(parts) if j != i])
            if len(fit_idx) < 100 or len(hold) < 30:
                continue
            w = self.tune_class_multipliers_raw(proba[fit_idx], y[fit_idx])
            sc_base = self.score(y[hold], np.argmax(proba[hold] * ones, axis=1))
            sc_new = self.score(y[hold], np.argmax(proba[hold] * w, axis=1))
            gains.append((sc_new - sc_base) if self.direction == "maximize" else (sc_base - sc_new))
        return float(np.mean(gains)) if gains else None

    # ------------------------------------------------------------------ #
    # final refit on the full training set
    # ------------------------------------------------------------------ #
    def refit_full(self, family: str, bundle, y: np.ndarray,
                   entity_mode: str = MODE_LOO, residual: bool = False,
                   seed: Optional[int] = None) -> Optional[Dict[str, Any]]:
        is_cls = self.profile.is_classification
        n_classes = int(self.profile.n_classes) if is_cls else 0
        X_tr = bundle.frame("train", None, mode=entity_mode)
        X_te = bundle.frame("test", None, mode=MODE_ALL)
        cats = bundle.cat_features_for(X_tr)
        rng = np.random.RandomState(self.seed if seed is None else seed)
        idx = rng.permutation(len(y))
        cut = max(30, int(0.92 * len(y)))
        tr_i, va_i = idx[:cut], idx[cut:]
        if residual and not is_cls and "ent_mean" in X_tr.columns:
            feats = X_tr.drop(columns=["ent_mean"])
            base_tr = X_tr["ent_mean"].values.astype(float)
            has = np.isfinite(base_tr)
            if has.sum() >= 100:
                tr_has = tr_i[has[tr_i]]
                out_a = self._fit_predict(family, feats.iloc[tr_has],
                                          y[tr_has] - base_tr[tr_has],
                                          feats.iloc[va_i[has[va_i]]],
                                          (y[va_i[has[va_i]]] - base_tr[va_i[has[va_i]]]),
                                          feats, cats, 0, seed=seed)
                if out_a is not None:
                    base_te = X_te["ent_mean"].values.astype(float)
                    has_te = np.isfinite(base_te)
                    test_pred = np.full(len(X_te), float(np.nanmedian(y)))
                    test_pred[has_te] = base_te[has_te] + out_a["test_pred"][has_te]
                    if (~has_te).any():
                        nohist = tr_i[~has[tr_i]]
                        if len(nohist) >= 100:
                            out_b = self._fit_predict(family, feats.iloc[nohist], y[nohist],
                                                      feats.iloc[va_i], y[va_i], feats, cats, 0,
                                                      seed=seed)
                            if out_b is not None:
                                test_pred[~has_te] = out_b["test_pred"][~has_te]
                    return {"test_pred": test_pred, "test_proba": None,
                            "best_iter": out_a.get("best_iter")}
        return self._fit_predict(family, X_tr.iloc[tr_i], y[tr_i], X_tr.iloc[va_i], y[va_i],
                                 X_te, cats, n_classes, seed=seed)


def _loglik(proba: np.ndarray, y: np.ndarray) -> float:
    proba = np.clip(np.asarray(proba, dtype=float), 1e-12, 1.0)
    n = len(y)
    return float(np.mean(np.log(proba[np.arange(n), np.asarray(y).astype(int)])))
