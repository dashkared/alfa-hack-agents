"""Agent 1 — TaskProfiler.

Deterministic-first experiment passport:
  * task type / target / id / metric detection from task.md + data_dictionary + CSV headers;
  * detection of a *temporal* column and of an *entity* (persistent object) key.

The platform LLM is only used as a bounded, optional fallback (one call, hard timeout);
every decision has a deterministic path so the pipeline can never depend on the network.
"""
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data_utils import (locate_file, optimize_dtypes, read_csv_safe,
                        split_feature_types)
from prompts import PROFILER_SYSTEM_PROMPT, TaskProfile, profiler_user_prompt

# --------------------------------------------------------------------------- #
# environment (LLM credentials may arrive via .env in the task directory)
# --------------------------------------------------------------------------- #
def load_env_fallback():
    """Load .env from the working directory or any of its parents (local development)."""
    candidates = [Path.cwd() / ".env", Path(__file__).resolve().parent / ".env"]
    for base in (Path.cwd(), Path(__file__).resolve().parent):
        parent = base
        for _ in range(4):
            parent = parent.parent
            candidates.append(parent / ".env")
    for env_path in candidates:
        try:
            if not env_path.exists():
                continue
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip("'\"")
                    if k and k not in os.environ:
                        os.environ[k] = v
            break
        except Exception:
            continue


load_env_fallback()


def build_llm_client(timeout: float = 60.0):
    """OpenAI-compatible client for the platform LLM (returns None when unavailable)."""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    base_url = (os.environ.get("OPENAI_BASE_URL") or "").strip() or None
    try:
        from openai import OpenAI
        return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=1)
    except Exception as exc:  # noqa: BLE001
        print(f"[TaskProfiler] LLM недоступен ({exc}); работаем детерминированно.")
        return None


# --------------------------------------------------------------------------- #
# keyword tables
# --------------------------------------------------------------------------- #
_REG_KW = ["rmse", "root mean squared", "среднеквадратичн", "mae", "mean absolute error",
           "r2", "r²", "mape", "rmsle", "medae", "регресс", "regression"]
_CLS_KW = ["macro f1", "macro-f1", "macro_f1", "f1-score", "f1_score", "f1", "accuracy",
           "точность", "roc-auc", "roc_auc", "auc", "log loss", "logloss", "balanced accuracy",
           "классификац", "classification"]

_TIME_NAME_HINTS = ["year", "date", "time", "period", "month", "quarter", "week", "day",
                    "год", "дата", "время", "период", "месяц", "квартал", "timestamp", "epoch"]


def _text(task_dir: Path) -> str:
    parts = []
    for name in ("task.md", "README.md", "readme.md"):
        try:
            p = locate_file(task_dir, name, required=False)
            if p is not None:
                parts.append(p.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
    return "\n".join(parts)


def _dictionary_text(task_dir: Path, max_rows: int = 25) -> str:
    try:
        p = locate_file(task_dir, "data_dictionary.csv", required=False)
        if p is None:
            return ""
        df = read_csv_safe(p)
        return df.head(max_rows).to_string()[:4000]
    except Exception:
        return ""


# --------------------------------------------------------------------------- #
# metric / task-type detection
# --------------------------------------------------------------------------- #
def detect_metric_from_text(text: str, is_classification: bool) -> Optional[str]:
    low = text.lower()
    if is_classification:
        order = [("macro f1", "Macro F1"), ("macro-f1", "Macro F1"), ("macro_f1", "Macro F1"),
                 ("f1-score", "Macro F1"), ("f1_score", "Macro F1"), ("weighted f1", "Weighted F1"),
                 ("balanced accuracy", "Balanced Accuracy"), ("roc-auc", "ROC-AUC"),
                 ("roc_auc", "ROC-AUC"), ("auc", "ROC-AUC"), ("log loss", "LogLoss"),
                 ("logloss", "LogLoss"), ("accuracy", "Accuracy")]
        for kw, name in order:
            if kw in low:
                return name
        return None
    order = [("rmse", "RMSE"), ("root mean squared", "RMSE"), ("среднеквадратичн", "RMSE"),
             ("mae", "MAE"), ("mean absolute error", "MAE"), ("rmsle", "RMSLE"),
             ("mape", "MAPE"), ("medae", "MedAE"), ("max error", "MaxError"),
             ("r2", "R2"), ("r²", "R2")]
    for kw, name in order:
        if kw in low:
            return name
    return None


def _looks_like_datetime(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_object_dtype(series):
        sample = series.dropna().astype(str).head(50)
        if len(sample) == 0:
            return False
        hits = sample.str.match(r"^\d{4}[-/.]\d{1,2}([-/.]\d{1,2})?([ T].*)?$").mean()
        return hits > 0.8
    return False


class TaskProfilerAgent:
    """Reads the task, inspects the tables and produces the `TaskProfile` passport."""

    def __init__(self, task_dir: Path, verbose: bool = True):
        self.task_dir = Path(task_dir).resolve()
        self.verbose = verbose

    # ------------------------------------------------------------------ #
    def run(self) -> Tuple[TaskProfile, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        train_path = locate_file(self.task_dir, "train.csv")
        test_path = locate_file(self.task_dir, "test.csv", required=False)
        sample_path = locate_file(self.task_dir, "sample_submission.csv", required=False)

        train_df = read_csv_safe(train_path)
        test_df = read_csv_safe(test_path) if test_path is not None else pd.DataFrame()
        sample_df = read_csv_safe(sample_path) if sample_path is not None else None
        # MLOps: mandatory in-memory compression before any modelling work
        try:
            train_df = optimize_dtypes(train_df, verbose=self.verbose)
            test_df = optimize_dtypes(test_df, verbose=self.verbose)
        except Exception:
            pass

        profile = TaskProfile(n_train=len(train_df), n_test=len(test_df))
        self._detect_id_target(profile, train_df, test_df, sample_df)
        self._detect_task_type(profile, train_df)
        self._finalise_metric(profile)
        self.detect_structure(profile, train_df, test_df)
        self._set_target_range(profile, train_df)
        return profile, train_df, test_df, sample_df

    # ------------------------------------------------------------------ #
    def _detect_id_target(self, profile: TaskProfile, train_df, test_df, sample_df):
        train_cols = list(train_df.columns)
        test_cols = list(test_df.columns) if len(test_df.columns) else []
        sample_cols = list(sample_df.columns) if sample_df is not None else []

        id_col = "id"
        if sample_cols:
            id_col = sample_cols[0]
        elif "id" in train_cols:
            id_col = "id"

        target_cols: List[str] = []
        if len(sample_cols) >= 2:
            target_cols = [c for c in sample_cols[1:] if c in train_cols]
        if not target_cols:
            diff = [c for c in train_cols if c not in test_cols and c != id_col]
            if diff:
                target_cols = [diff[-1]]
        if not target_cols:
            target_cols = [train_cols[-1]]

        profile.id_column = id_col
        profile.target_columns = target_cols
        profile.target_column = target_cols[0]
        if len(target_cols) > 1:
            profile.task_type = "multilabel"
            profile.notes["multilabel_columns"] = target_cols

    def _detect_task_type(self, profile: TaskProfile, train_df: pd.DataFrame):
        if profile.task_type == "multilabel":
            profile.metric = "Macro F1"
            profile.objective = "maximize"
            profile.class_labels = []
            return
        col = profile.target_column
        if col not in train_df.columns:
            profile.task_type = "regression"
            return
        try:
            s = train_df[col]
            s_num = pd.to_numeric(s, errors="coerce")
            numeric_ratio = float(s_num.notna().mean())
            n = max(len(s), 1)
            nuniq = int(s.nunique(dropna=True))
            is_integer = bool((s_num.dropna() % 1 == 0).all()) if numeric_ratio > 0 else False
            if numeric_ratio < 0.98:
                profile.task_type = "binary" if nuniq <= 2 else "multiclass"
            elif nuniq <= 2:
                profile.task_type = "binary"
            elif is_integer and nuniq <= 20 and nuniq / n <= 0.02:
                profile.task_type = "multiclass"
            else:
                profile.task_type = "regression"
        except Exception:
            profile.task_type = "regression"

        # the task text is a strong prior for integer-coded targets
        text = _text(self.task_dir).lower()
        has_reg = any(k in text for k in _REG_KW)
        has_cls = any(k in text for k in _CLS_KW)
        metric_reg = detect_metric_from_text(text, False)
        metric_cls = detect_metric_from_text(text, True)
        try:
            nuniq = int(train_df[col].nunique(dropna=True)) if col in train_df.columns else 0
        except Exception:
            nuniq = 0
        if profile.task_type == "regression" and metric_cls and not metric_reg and 2 < nuniq <= 30:
            profile.task_type = "multiclass"
        elif profile.task_type == "regression" and metric_cls and not metric_reg and nuniq == 2:
            profile.task_type = "binary"
        elif profile.task_type in ("binary", "multiclass") and metric_reg and not metric_cls \
                and nuniq > 30:
            profile.task_type = "regression"
        if profile.is_classification:
            try:
                labels = sorted({str(v) for v in train_df[col].dropna().unique()})
                profile.class_labels = labels
                profile.n_classes = len(labels)
            except Exception:
                profile.class_labels, profile.n_classes = [], 0
        profile.notes["text_mentions_regression"] = has_reg
        profile.notes["text_mentions_classification"] = has_cls

    def _finalise_metric(self, profile: TaskProfile):
        text = _text(self.task_dir)
        metric = detect_metric_from_text(text, profile.is_classification)
        if profile.task_type == "multilabel":
            profile.metric, profile.objective = "Macro F1", "maximize"
            return
        if metric is None:
            metric = "RMSE" if profile.is_regression else ("Macro F1" if profile.task_type == "multiclass"
                                                           else "Macro F1")
        profile.metric = metric
        profile.objective = "maximize" if metric.lower() in (
            "macro f1", "weighted f1", "accuracy", "balanced accuracy", "roc-auc", "auc") else "minimize"
        # binary: submit probabilities when the metric is ranking-like
        if profile.task_type == "binary":
            profile.predict_proba = metric.lower() in ("roc-auc", "auc", "logloss", "log loss")

    # ------------------------------------------------------------------ #
    # structure detection: temporal column and entity key
    # ------------------------------------------------------------------ #
    def detect_structure(self, profile: TaskProfile, train_df: pd.DataFrame, test_df: pd.DataFrame):
        if len(test_df) == 0:
            return
        common = [c for c in train_df.columns if c in test_df.columns]
        excluded = {profile.id_column, profile.target_column} | set(profile.target_columns)
        candidates = [c for c in common if c not in excluded]

        time_col = self._detect_time_column(train_df, test_df, candidates)
        if time_col is not None:
            profile.time_column = time_col
            try:
                tr_vals = train_df[time_col].dropna().unique()
                te_vals = test_df[time_col].dropna().unique()
                tr_sorted = sorted(tr_vals)
                profile.time_periods = tr_sorted
                profile.test_periods = sorted(te_vals)
                numeric = pd.api.types.is_numeric_dtype(train_df[time_col])
                if numeric and len(profile.test_periods) and len(tr_sorted):
                    profile.temporal_extrapolation = bool(
                        min(profile.test_periods) > max(tr_sorted))
                else:
                    profile.temporal_extrapolation = bool(
                        len(set(profile.test_periods) - set(tr_sorted)) > 0)
            except Exception:
                profile.time_column = None

        ent_cols, score, coverage = self._discover_entity_columns(
            profile, train_df, test_df,
            candidates=[c for c in candidates if c != profile.time_column])
        if ent_cols:
            profile.entity_columns = ent_cols
            profile.entity_score = score
            profile.entity_coverage = coverage

    def _detect_time_column(self, train_df, test_df, candidates) -> Optional[str]:
        best, best_score = None, 0.0
        n_tr = max(len(train_df), 1)
        for col in candidates:
            try:
                s_tr, s_te = train_df[col], test_df[col]
                if _looks_like_datetime(s_tr):
                    score = 1.2
                    if any(h in col.lower() for h in _TIME_NAME_HINTS):
                        score += 0.3
                    if score > best_score:
                        best, best_score = col, score
                    continue
                u_tr = pd.unique(s_tr.dropna())
                u_te = pd.unique(s_te.dropna())
                if len(u_tr) < 2 or len(u_te) == 0 or len(u_tr) > 120:
                    continue
                if pd.api.types.is_float_dtype(s_tr) and s_tr.nunique() > 60:
                    continue
                score = 0.0
                numeric = pd.api.types.is_numeric_dtype(s_tr)
                if numeric:
                    try:
                        if min(u_te) > max(u_tr):
                            score += 0.8          # pure future extrapolation
                        elif len(set(u_te) & set(u_tr)) == 0:
                            score += 0.6
                    except Exception:
                        pass
                else:
                    inter = set(map(str, u_te)) & set(map(str, u_tr))
                    if len(inter) == 0:
                        score += 0.6
                if score == 0.0:
                    continue
                if any(h in col.lower() for h in _TIME_NAME_HINTS):
                    score += 0.3
                if len(u_te) <= 3:
                    score += 0.2
                if len(u_tr) / n_tr > 0.2:            # a period column is low cardinality
                    score -= 0.5
                if score > best_score:
                    best, best_score = col, score
            except Exception:
                continue
        if best_score >= 0.9:
            return best
        return None

    # ------------------------------------------------------------------ #
    def _discover_entity_columns(self, profile: TaskProfile, train_df: pd.DataFrame,
                                 test_df: pd.DataFrame, candidates: List[str],
                                 max_cols: int = 4, max_candidates: int = 100,
                                 min_single_j: float = -1.0
                                 ) -> Tuple[List[str], float, float]:
        """Greedy search for a composite key identifying a persistent object.

        J(S) = coverage_test(S) * (1 - loo_error_ratio(S)) - parsimony.
        A key is accepted only if it explains a large share of the target variance, is
        covered in the test split and is *repeating* (many rows really have a sibling).
        """
        if not candidates or profile.task_type == "multilabel":
            return [], 0.0, 0.0
        target = profile.target_column
        try:
            y_raw = train_df[target]
            if profile.is_classification:
                codes, _ = pd.factorize(y_raw.astype(str), sort=True)
                y = codes.astype(float)
                counts = np.bincount(codes)
                baseline_err = max(1e-6, 1.0 - counts.max() / max(counts.sum(), 1))
            else:
                y_num = pd.to_numeric(y_raw, errors="coerce").astype(float)
                finite = np.isfinite(y_num)
                if finite.mean() < 0.8:
                    return [], 0.0, 0.0
                y = np.where(finite, y_num, np.nan)
                baseline_err = float(np.nanstd(y))
            if not np.isfinite(baseline_err) or baseline_err <= 0:
                return [], 0.0, 0.0
        except Exception:
            return [], 0.0, 0.0

        n_tr, n_te = len(train_df), len(test_df)
        if n_te == 0:
            return [], 0.0, 0.0
        combined = pd.concat([train_df[candidates], test_df[candidates]], ignore_index=True)

        cache: Dict[str, np.ndarray] = {}

        def col_codes(col: str) -> np.ndarray:
            if col not in cache:
                codes, _ = pd.factorize(combined[col].astype(str).values, sort=False)
                codes = codes.copy()
                codes[codes < 0] = codes.max() + 1
                cache[col] = codes.astype(np.int64)
            return cache[col]

        def key_codes(cols: List[str]) -> np.ndarray:
            if len(cols) == 1:
                return col_codes(cols[0])
            mat = np.stack([col_codes(c) for c in cols], axis=1)
            _, inv = np.unique(mat, axis=0, return_inverse=True)
            return inv.astype(np.int64)

        pool: List[str] = []
        for col in candidates:
            try:
                nuniq = int(combined[col].nunique(dropna=True))
                if nuniq < 2:
                    continue
                if pd.api.types.is_float_dtype(combined[col]) and nuniq > 0.6 * (n_tr + n_te):
                    frac_non_int = float((combined[col].dropna() % 1 != 0).mean())
                    if frac_non_int > 0.2:
                        continue
                pool.append(col)
            except Exception:
                continue
        if not pool:
            return [], 0.0, 0.0

        def stats(cols: List[str]) -> Tuple[float, float, float]:
            """(J, coverage, hist_frac)"""
            codes = key_codes(cols)
            tr_codes, te_codes = codes[:n_tr], codes[n_tr:]
            n_groups = int(codes.max()) + 1
            cnt = np.bincount(tr_codes, minlength=n_groups).astype(float)
            coverage = float(np.isin(te_codes, tr_codes).mean())
            hist_frac = float(np.mean(cnt[tr_codes] - 1 >= 1))
            q = self._loo_quality_codes(tr_codes, y, baseline_err, profile.is_classification)
            return coverage * q - 0.01 * len(cols), coverage, hist_frac

        singles = []
        for col in pool:
            try:
                j, cov, hf = stats([col])
                singles.append((j, col))
            except Exception:
                continue
        singles.sort(reverse=True)
        if not singles:
            return [], 0.0, 0.0
        pool = [c for j, c in singles[:max_candidates] if j > min_single_j]
        if not pool:
            return [], 0.0, 0.0

        chosen: List[str] = []
        current_j = 0.0
        best_cov, best_hf = 0.0, 0.0
        for _ in range(max_cols):
            trials = []
            for col in [c for c in pool if c not in chosen]:
                try:
                    j, cov, hf = stats(chosen + [col])
                    trials.append((j, col, cov, hf))
                except Exception:
                    continue
            if not trials:
                break
            trials.sort(reverse=True)
            j, col, cov, hf = trials[0]
            if j <= current_j + 0.005:
                break
            chosen.append(col)
            current_j, best_cov, best_hf = j, cov, hf
        if not chosen:
            return [], 0.0, 0.0
        j, coverage, hist_frac = stats(chosen)
        if j < 0.10 or coverage < 0.20 or hist_frac < 0.30:
            return [], float(j), coverage
        profile.notes["entity_hist_frac"] = hist_frac
        return chosen, float(j), coverage

    @staticmethod
    def _loo_quality_codes(codes: np.ndarray, y: np.ndarray, baseline_err: float,
                           is_classification: bool) -> float:
        """Leave-one-out group prediction error reduction, given pre-factorised codes."""
        if len(codes) == 0:
            return 0.0
        n_groups = int(codes.max()) + 1
        if is_classification:
            yi = y.astype(np.int64)
            n_classes = int(yi.max()) + 1
            sums = np.bincount(codes * n_classes + yi,
                               minlength=n_groups * n_classes).reshape(n_groups, n_classes)
            cnt = sums.sum(axis=1)
            loo = sums[codes].copy()
            loo[np.arange(len(codes)), yi] -= 1
            valid = (cnt[codes] - 1) >= 1
            if valid.sum() < 20:
                return 0.0
            pred = loo.argmax(axis=1)
            err = float(np.mean(pred[valid] != yi[valid]))
            return float(max(0.0, 1.0 - err / max(baseline_err, 1e-9)))
        yv = np.asarray(y, dtype=float)
        finite = np.isfinite(yv)
        if finite.sum() < 20:
            return 0.0
        c_f, y_f = codes[finite], yv[finite]
        sums = np.bincount(c_f, weights=y_f, minlength=n_groups)
        cnt = np.bincount(c_f, minlength=n_groups).astype(float)
        cnt_row = cnt[c_f]
        loo_mean = (sums[c_f] - y_f) / np.maximum(cnt_row - 1, 1)
        valid = cnt_row >= 2
        if valid.sum() < 20:
            return 0.0
        err = float(np.sqrt(np.mean((y_f[valid] - loo_mean[valid]) ** 2)))
        return float(max(0.0, 1.0 - min(err / max(baseline_err, 1e-9), 1.0)))

    @staticmethod
    def _loo_quality(keys: pd.Series, y: np.ndarray, baseline_err: float,
                     is_classification: bool) -> float:
        """Error reduction in [0, 1] of a leave-one-out group prediction (vectorised)."""
        codes, _ = pd.factorize(keys.astype(str).values, sort=False)
        if len(codes) == 0 or (codes < 0).all():
            return 0.0
        nogroup = codes < 0
        if nogroup.any():
            codes = codes.copy()
            codes[nogroup] = codes.max() + 1
        n_groups = int(codes.max()) + 1
        if is_classification:
            yi = y.astype(np.int64)
            n_classes = int(yi.max()) + 1
            flat = np.bincount(codes * n_classes + yi, minlength=n_groups * n_classes)
            sums = flat.reshape(n_groups, n_classes)
            cnt = sums.sum(axis=1)
            loo = sums[codes].copy()
            loo[np.arange(len(codes)), yi] -= 1
            valid = (cnt[codes] - 1) >= 1
            if valid.sum() < 20:
                return 0.0
            pred = loo.argmax(axis=1)
            err = float(np.mean(pred[valid] != yi[valid]))
            return float(max(0.0, 1.0 - err / max(baseline_err, 1e-9)))
        yv = np.asarray(y, dtype=float)
        finite = np.isfinite(yv)
        if finite.sum() < 20:
            return 0.0
        codes_f = codes[finite]
        yf = yv[finite]
        sums = np.bincount(codes_f, weights=yf, minlength=n_groups)
        cnt = np.bincount(codes_f, minlength=n_groups).astype(float)
        cnt_row = cnt[codes_f]
        loo_mean = (sums[codes_f] - yf) / np.maximum(cnt_row - 1, 1)
        valid = (cnt_row >= 2)
        if valid.sum() < 20:
            return 0.0
        err = float(np.sqrt(np.mean((yf[valid] - loo_mean[valid]) ** 2)))
        ratio = err / max(baseline_err, 1e-9)
        return float(max(0.0, 1.0 - min(ratio, 1.0)))

    # ------------------------------------------------------------------ #
    def _set_target_range(self, profile: TaskProfile, train_df: pd.DataFrame):
        if not profile.is_regression:
            return
        try:
            y = pd.to_numeric(train_df[profile.target_column], errors="coerce").dropna()
            if len(y):
                profile.target_min = float(y.min())
                profile.target_max = float(y.max())
                profile.notes["target_p01"] = float(y.quantile(0.001))
                profile.notes["target_p99"] = float(y.quantile(0.999))
                profile.notes["target_skew"] = float(y.skew())
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    def llm_assist(self, profile: TaskProfile, train_df: pd.DataFrame) -> TaskProfile:
        """Optional bounded LLM refinement of an ambiguous passport."""
        client = build_llm_client(timeout=60.0)
        if client is None:
            return profile
        model = os.environ.get("OPENAI_MODEL", "qwen/qwen3.8-27b")
        overview = (
            f"колонки train ({len(train_df.columns)}): {list(train_df.columns)[:40]}\n"
            f"первые 3 строки:\n{train_df.head(3).to_string()[:1200]}\n"
            f"task_type по эвристике: {profile.task_type}, метрика: {profile.metric}"
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": PROFILER_SYSTEM_PROMPT},
                          {"role": "user", "content": profiler_user_prompt(
                              _text(self.task_dir), _dictionary_text(self.task_dir), overview)}],
                response_format={"type": "json_object"},
                temperature=0.0, max_tokens=600,
            )
            content = resp.choices[0].message.content
            if not content:
                return profile
            data = json.loads(content)
            tgt = str(data.get("target") or "").strip()
            metric = str(data.get("metric") or "").strip()
            valid_targets = set(train_df.columns)
            if tgt in valid_targets and tgt != profile.id_column:
                candidate_types = {"regression"} if profile.is_regression else {"binary", "multiclass"}
                if str(data.get("task_type")) in candidate_types:
                    profile.target_column = tgt
            if metric:
                low = metric.lower()
                if (profile.is_regression and low in ("rmse", "mae", "r2", "mape", "rmsle")) or \
                   (profile.is_classification and low in ("macro f1", "weighted f1", "roc-auc",
                                                          "accuracy", "logloss", "log loss",
                                                          "balanced accuracy")):
                    profile.metric = metric
                    profile.objective = "maximize" if low in (
                        "macro f1", "weighted f1", "roc-auc", "accuracy", "balanced accuracy") else "minimize"
            profile.notes["llm_used"] = True
            if self.verbose:
                print(f"[TaskProfiler] LLM уточнил паспорт: metric={profile.metric}, "
                      f"target={profile.target_column}")
        except Exception as exc:  # noqa: BLE001
            if self.verbose:
                print(f"[TaskProfiler] LLM-уточнение пропущено ({type(exc).__name__}).")
        return profile


# --------------------------------------------------------------------------- #
# lightweight compatibility wrapper (used by the emergency recovery path)
# --------------------------------------------------------------------------- #
def heuristic_task_profile(task_dir: Path) -> TaskProfile:
    profiler = TaskProfilerAgent(Path(task_dir), verbose=False)
    try:
        profile, _, _, _ = profiler.run()
        return profile
    except Exception:
        return TaskProfile()


def standardize_target(series: pd.Series, profile: TaskProfile):
    """Model-space target encoding.  Returns (y_model, class_labels)."""
    if profile.is_regression:
        y = pd.to_numeric(series, errors="coerce").astype(float)
        fill = float(np.nanmedian(y)) if np.isfinite(np.nanmedian(y)) else 0.0
        y = y.fillna(fill)
        if profile.target_transform == "log1p" and (y >= 0).all():
            return np.log1p(y.values), None
        return y.values, None
    labels = profile.class_labels if profile.class_labels else \
        sorted({str(v) for v in series.dropna().unique()})
    mapping = {lab: i for i, lab in enumerate(labels)}
    codes = series.astype(str).map(mapping)
    if codes.isna().any():
        codes = codes.fillna(0)
    return codes.astype(int).values, labels


def inverse_target(pred_model: np.ndarray, profile: TaskProfile, labels: Optional[List[str]]):
    """Back-transform model-space predictions into submission space."""
    arr = np.asarray(pred_model)
    if profile.is_regression:
        if profile.target_transform == "log1p":
            arr = np.expm1(arr)
        if profile.target_min is not None or profile.target_max is not None:
            lo = profile.target_min if profile.target_min is not None else -np.inf
            hi = profile.target_max if profile.target_max is not None else np.inf
            arr = np.clip(arr, lo, hi)
        return arr
    labels = labels or profile.class_labels
    if profile.predict_proba and arr.ndim == 2:
        return arr[:, 1] if arr.shape[1] == 2 else arr
    if arr.ndim == 2:
        arr = np.argmax(arr, axis=1)
    arr = np.asarray(arr, dtype=int)
    if labels:
        labels_arr = np.asarray(labels, dtype=object)
        arr = np.clip(arr, 0, len(labels_arr) - 1)
        return labels_arr[arr]
    return arr
