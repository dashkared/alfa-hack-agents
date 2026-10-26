"""Data IO, dtype compression and bullet-proof submission writing.

Design rules (learned the hard way):
  * never rely on pandas-3-only behaviour; the evaluation container runs pandas 2.2;
  * object columns are kept as `object` (never silently cast to `category`), because
    downstream `.map(dict)` / `.fillna()` chains behave differently for Categorical;
  * numeric downcasting is done column-wise with numpy bounds checks;
  * `submission.csv` is written atomically and validated against `sample_submission.csv`.
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

FORBIDDEN_TARGET_NAMES = {"id", "index"}


# --------------------------------------------------------------------------- #
# File discovery
# --------------------------------------------------------------------------- #
def locate_file(task_dir: Path, filename: str, required: bool = True) -> Optional[Path]:
    """Find `filename` in the task directory (or one level of sub-directories)."""
    task_dir = Path(task_dir).resolve()
    direct = task_dir / filename
    if direct.is_file():
        return direct
    # nested layout, e.g. task.md references `regression/participants/train.csv`
    if task_dir.is_dir():
        matches = sorted(task_dir.rglob(filename))
        if matches:
            return matches[0]
    cwd = Path.cwd().resolve()
    if (cwd / filename).is_file():
        return cwd / filename
    if required:
        raise FileNotFoundError(f"Не найден обязательный файл '{filename}' в {task_dir}")
    return None


def read_csv_safe(path: Path, **kwargs) -> pd.DataFrame:
    """Read a CSV with a couple of fallbacks for odd encodings/separators."""
    path = Path(path)
    attempts = [
        dict(sep=",", encoding="utf-8"),
        dict(sep=",", encoding="utf-8-sig"),
        dict(sep=";", encoding="utf-8"),
        dict(sep=None, engine="python", encoding="utf-8"),
        dict(sep=",", encoding="latin-1"),
    ]
    last_err = None
    for kw in attempts:
        try:
            df = pd.read_csv(path, low_memory=False, **{**kw, **kwargs})
            if df.shape[1] >= 1 and len(df) > 0:
                return df
        except Exception as e:  # noqa: BLE001
            last_err = e
    if last_err is not None:
        raise last_err
    return pd.read_csv(path, **kwargs)


# --------------------------------------------------------------------------- #
# dtype compression (platform requirement: downcasting on step 1)
# --------------------------------------------------------------------------- #
def optimize_dtypes(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Downcast numeric columns.  Object columns are intentionally left untouched."""
    if df.empty:
        return df
    start_mb = df.memory_usage(deep=True).sum() / 1e6
    for col in df.columns:
        dt = df[col].dtype
        try:
            if pd.api.types.is_integer_dtype(dt):
                c_min, c_max = df[col].min(), df[col].max()
                if pd.isna(c_min) or pd.isna(c_max):
                    continue
                for typ in (np.int8, np.int16, np.int32):
                    if c_min >= np.iinfo(typ).min and c_max <= np.iinfo(typ).max:
                        df[col] = df[col].astype(typ)
                        break
                else:
                    df[col] = df[col].astype(np.int64)
            elif pd.api.types.is_float_dtype(dt):
                if dt != np.float32:
                    df[col] = df[col].astype(np.float32)
            elif pd.api.types.is_bool_dtype(dt):
                df[col] = df[col].astype(np.int8)
        except Exception:
            continue
    if verbose:
        end_mb = df.memory_usage(deep=True).sum() / 1e6
        print(f"[DataCompressor] Память: {start_mb:.1f} MB -> {end_mb:.1f} MB")
    return df


def split_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return (cat_features, num_features) using dtype + cardinality heuristics."""
    cat_cols, num_cols = [], []
    n = max(len(X), 1)
    for col in X.columns:
        dt = X[col].dtype
        if pd.api.types.is_bool_dtype(dt) or pd.api.types.is_object_dtype(dt) \
                or isinstance(dt, pd.CategoricalDtype) or pd.api.types.is_string_dtype(dt):
            cat_cols.append(col)
        elif pd.api.types.is_numeric_dtype(dt):
            num_cols.append(col)
        else:  # datetime & friends -> treated as categorical after stringification
            try:
                X[col] = X[col].astype(str)
            except Exception:
                pass
            cat_cols.append(col)
    return cat_cols, num_cols


# --------------------------------------------------------------------------- #
# Submission handling
# --------------------------------------------------------------------------- #
def make_submission_frame(test_ids: pd.Series, predictions, profile) -> pd.DataFrame:
    """Build a submission DataFrame matching the expected column names."""
    preds = np.asarray(predictions)
    if preds.ndim > 1:
        preds = preds.ravel()
    target_name = getattr(profile, "target_column", "target") or "target"
    id_name = getattr(profile, "id_column", "id") or "id"
    df = pd.DataFrame({id_name: np.asarray(test_ids), target_name: preds})
    return df


def _align_with_sample(df: pd.DataFrame, sample: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Match column names/dtypes and row order with sample_submission when possible."""
    if sample is None or sample.shape[1] != 2 or df.shape[1] != 2:
        return df
    df = df.copy()
    df.columns = list(sample.columns)
    sample_id_col, sample_tgt_col = sample.columns[0], sample.columns[1]
    # row order: align on id if the sample ids are a permutation of ours
    try:
        if len(sample) == len(df):
            sample_ids = sample[sample_id_col]
            our_ids = df[sample_id_col]
            if not np.array_equal(np.asarray(sample_ids), np.asarray(our_ids)):
                tmp = df.set_index(sample_id_col, drop=False)
                if tmp.index.is_unique and sample_ids.isin(tmp.index).all():
                    df = tmp.loc[np.asarray(sample_ids)].reset_index(drop=True)
                    df.columns = list(sample.columns)
    except Exception:
        pass
    # id dtype: keep sample dtype when it round-trips
    try:
        target_dt = sample_id_col and sample[sample_id_col].dtype
        if pd.api.types.is_numeric_dtype(target_dt):
            df[sample_id_col] = pd.to_numeric(df[sample_id_col], errors="coerce").fillna(-1)
            df[sample_id_col] = df[sample_id_col].astype(target_dt)
    except Exception:
        pass
    return df


def _sanitize_predictions(df: pd.DataFrame, profile) -> pd.DataFrame:
    """Guarantee finite values and correct dtypes in the prediction column."""
    tgt = df.columns[-1]
    task_type = getattr(profile, "task_type", "regression")
    if task_type == "regression":
        vals = pd.to_numeric(df[tgt], errors="coerce")
        arr = np.asarray(vals, dtype=float)
        finite = np.isfinite(arr)
        if not finite.all():
            fill = float(np.nanmedian(arr[finite])) if finite.any() else 0.0
            arr = np.where(finite, arr, fill)
        df[tgt] = arr.astype(np.float64)
    else:
        if getattr(profile, "predict_proba", False):
            # probability submissions (e.g. ROC-AUC metrics): keep numeric scores
            vals = pd.to_numeric(df[tgt], errors="coerce")
            arr = np.asarray(vals, dtype=float)
            finite = np.isfinite(arr)
            if not finite.all():
                fill = float(np.nanmedian(arr[finite])) if finite.any() else 0.0
                arr = np.where(finite, arr, fill)
            df[tgt] = arr
            return df
        vals = df[tgt].astype(str)
        labels = getattr(profile, "class_labels", None)
        if labels:
            bad = ~vals.isin([str(v) for v in labels])
            if bad.any():
                vals = vals.where(~bad, str(labels[0]))
        df[tgt] = vals
    return df


def save_submission(task_dir: Path, test_ids: pd.Series, predictions, profile) -> Optional[Path]:
    """Atomic submission writer.  Always leaves a readable `submission.csv` behind."""
    task_dir = Path(task_dir).resolve()
    df = make_submission_frame(test_ids, predictions, profile)
    sample = None
    try:
        sample_path = locate_file(task_dir, "sample_submission.csv", required=False)
        if sample_path is not None:
            sample = read_csv_safe(sample_path, nrows=max(len(df) + 10, 10))
    except Exception:
        sample = None
    df = _align_with_sample(df, sample)
    df = _sanitize_predictions(df, profile)

    targets = [task_dir]
    try:
        cwd = Path.cwd().resolve()
        if cwd != task_dir:
            targets.append(cwd)
    except Exception:
        pass
    written = None
    for out_dir in targets:
        try:
            tmp = out_dir / f".submission_tmp_{os.getpid()}_{int(np.random.randint(1e9))}.csv"
            df.to_csv(tmp, index=False, encoding="utf-8")
            final = out_dir / "submission.csv"
            os.replace(tmp, final)
            written = final if written is None else written
        except Exception:
            try:
                final = out_dir / "submission.csv"
                df.to_csv(final, index=False, encoding="utf-8")
                written = final if written is None else written
            except Exception:
                pass
    return written


def validate_submission(task_dir: Path, profile, n_expected: Optional[int] = None) -> Tuple[bool, str]:
    """Final gate: check the written submission against the sample submission."""
    task_dir = Path(task_dir).resolve()
    sub_path = task_dir / "submission.csv"
    if not sub_path.is_file() or sub_path.stat().st_size == 0:
        return False, "submission.csv отсутствует или пуст"
    try:
        sub = read_csv_safe(sub_path)
    except Exception as e:  # noqa: BLE001
        return False, f"submission.csv не читается: {e}"
    sample_path = locate_file(task_dir, "sample_submission.csv", required=False)
    if sample_path is not None:
        try:
            sample = read_csv_safe(sample_path)
            if list(sub.columns) != list(sample.columns):
                return False, f"колонки {list(sub.columns)} != {list(sample.columns)}"
            if len(sub) != len(sample):
                return False, f"строк {len(sub)} != {len(sample)}"
            id_col = sample.columns[0]
            if not np.array_equal(np.asarray(sub[id_col]), np.asarray(sample[id_col])):
                return False, "порядок/значения id не совпадают с sample_submission"
        except Exception as e:  # noqa: BLE001
            return False, f"сверка с sample_submission не удалась: {e}"
    if n_expected is not None and len(sub) != n_expected:
        return False, f"строк {len(sub)} != ожидаемых {n_expected}"
    tgt = sub.columns[-1]
    if sub[tgt].isnull().any():
        return False, f"в колонке {tgt} есть пропуски"
    if getattr(profile, "task_type", "regression") == "regression":
        vals = pd.to_numeric(sub[tgt], errors="coerce")
        if vals.isnull().any() or not np.isfinite(np.asarray(vals, float)).all():
            return False, "регрессионные предсказания содержат NaN/inf"
    return True, f"OK: {len(sub)} строк, колонки {list(sub.columns)}"
