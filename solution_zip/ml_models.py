from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from data_utils import save_atomic_submission
from prompts import TaskProfile
from sklearn.metrics import f1_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder


def generate_failsafe_baseline(
    y_train: pd.Series,
    test_ids: pd.Series,
    profile: TaskProfile,
    task_dir: Any,
) -> np.ndarray:
    """Мгновенный бейзлайн (Fail-Safe), сохраняющий гарантированный сабмит на 1-й минуте."""
    print("[MLModels] Генерация защитного бейзлайна (Fail-Safe)...")

    if profile.task_type == "regression":
        median_val = float(y_train.median())
        preds = np.full(len(test_ids), median_val, dtype=np.float32)
    else:
        mode_val = y_train.mode()[0]
        preds = np.full(len(test_ids), mode_val)

    save_atomic_submission(task_dir, test_ids, preds, profile)
    print(f"[MLModels] Fail-Safe сабмит сохранен: значение по умолчанию = {preds[0]}")
    return preds


def train_gradient_boosting_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    cat_features: List[str],
    num_features: List[str],
    profile: TaskProfile,
    n_splits: int = 5,
) -> Tuple[float, np.ndarray, str]:
    """Обучение градиентного бустинга с 5-Fold кросс-валидацией и оценкой OOF-метрик."""
    print(f"[MLModels] Запуск 5-Fold CV обучения для задачи: {profile.task_type} (Метрика: {profile.metric})...")

    # Предобработка категориальных признаков через Label Encoding для универсальности
    X_train_enc = X_train.copy()
    X_test_enc = X_test.copy()

    for col in cat_features:
        le = LabelEncoder()
        # Объединяем уникальные значения
        combined = pd.concat([X_train_enc[col], X_test_enc[col]], axis=0).astype(str)
        le.fit(combined)
        X_train_enc[col] = le.transform(X_train_enc[col].astype(str))
        X_test_enc[col] = le.transform(X_test_enc[col].astype(str))

    # Выбор бэкенда: CatBoost -> LightGBM -> HistGradientBoosting (Sklearn)
    backend = "catboost"
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
    except Exception:
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor
            backend = "lightgbm"
        except Exception:
            from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
            backend = "hist_gb"

    print(f"[MLModels] Используемый бэкенд модели: {backend}")

    # =========================================================================
    # 1. РЕГРЕССИЯ
    # =========================================================================
    if profile.task_type == "regression":
        y_vals = y_train.values.astype(np.float32)
        oof_preds = np.zeros(len(y_vals), dtype=np.float32)
        test_preds = np.zeros(len(X_test_enc), dtype=np.float32)

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_enc, y_vals)):
            X_tr, y_tr = X_train_enc.iloc[train_idx], y_vals[train_idx]
            X_va, y_va = X_train_enc.iloc[val_idx], y_vals[val_idx]

            if backend == "catboost":
                from catboost import CatBoostRegressor
                model = CatBoostRegressor(
                    iterations=600,
                    learning_rate=0.08,
                    depth=6,
                    thread_count=8,
                    random_seed=42 + fold,
                    verbose=0,
                )
                model.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50, verbose=False)
            elif backend == "lightgbm":
                from lightgbm import LGBMRegressor
                model = LGBMRegressor(
                    n_estimators=600,
                    learning_rate=0.08,
                    max_depth=6,
                    n_jobs=8,
                    random_state=42 + fold,
                    verbose=-1,
                )
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
            else:
                from sklearn.ensemble import HistGradientBoostingRegressor
                model = HistGradientBoostingRegressor(
                    max_iter=300,
                    learning_rate=0.08,
                    max_depth=8,
                    random_state=42 + fold,
                )
                model.fit(X_tr, y_tr)

            val_preds = model.predict(X_va)
            oof_preds[val_idx] = val_preds
            test_preds += model.predict(X_test_enc) / n_splits

        # Расчет RMSE
        oof_score = float(np.sqrt(mean_squared_error(y_vals, oof_preds)))
        print(f"[MLModels] 5-Fold OOF RMSE: {oof_score:.4f}")
        return oof_score, test_preds, backend

    # =========================================================================
    # 2. КЛАССИФИКАЦИЯ (Бинарная или Многоклассовая)
    # =========================================================================
    else:
        label_enc = LabelEncoder()
        y_encoded = label_enc.fit_transform(y_train.astype(str))
        num_classes = len(label_enc.classes_)
        oof_probs = np.zeros((len(y_encoded), num_classes), dtype=np.float32)
        test_probs = np.zeros((len(X_test_enc), num_classes), dtype=np.float32)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_enc, y_encoded)):
            X_tr, y_tr = X_train_enc.iloc[train_idx], y_encoded[train_idx]
            X_va, y_va = X_train_enc.iloc[val_idx], y_encoded[val_idx]

            if backend == "catboost":
                from catboost import CatBoostClassifier
                model = CatBoostClassifier(
                    iterations=600,
                    learning_rate=0.08,
                    depth=6,
                    loss_function="MultiClass" if num_classes > 2 else "Logloss",
                    auto_class_weights="Balanced",
                    thread_count=8,
                    random_seed=42 + fold,
                    verbose=0,
                )
                model.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50, verbose=False)
                val_p = model.predict_proba(X_va)
                t_p = model.predict_proba(X_test_enc)
            elif backend == "lightgbm":
                from lightgbm import LGBMClassifier
                model = LGBMClassifier(
                    n_estimators=600,
                    learning_rate=0.08,
                    max_depth=6,
                    class_weight="balanced",
                    n_jobs=8,
                    random_state=42 + fold,
                    verbose=-1,
                )
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
                val_p = model.predict_proba(X_va)
                t_p = model.predict_proba(X_test_enc)
            else:
                from sklearn.ensemble import HistGradientBoostingClassifier
                model = HistGradientBoostingClassifier(
                    max_iter=300,
                    learning_rate=0.08,
                    max_depth=8,
                    class_weight="balanced",
                    random_state=42 + fold,
                )
                model.fit(X_tr, y_tr)
                val_p = model.predict_proba(X_va)
                t_p = model.predict_proba(X_test_enc)

            oof_probs[val_idx] = val_p
            test_probs += t_p / n_splits

        oof_pred_labels = np.argmax(oof_probs, axis=1)
        test_pred_labels = np.argmax(test_probs, axis=1)

        # Переводим обратно в исходные строковые имена классов
        final_test_preds = label_enc.inverse_transform(test_pred_labels)

        oof_score = float(f1_score(y_encoded, oof_pred_labels, average="macro"))
        print(f"[MLModels] 5-Fold OOF Macro F1: {oof_score:.4f}")
        return oof_score, final_test_preds, backend
