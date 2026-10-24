import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from prompts import TaskProfile
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


class ModelPortfolioAgent:
    """Агент 3: ML-Архитектор.

    Управляет портфелем моделей «Большой тройки» (CatBoost, LightGBM, XGBoost),
    выполняет очистку неинформативных признаков (Feature Pruning),
    оптимизирует многомерные пороги для редких классов и строит оптимальные ансамбли (OOF Blending и Multi-Layer Stacking).
    """

    def __init__(self, profile: TaskProfile, n_splits: int = 5):
        self.profile = profile
        self.n_splits = n_splits
        # Пул обученных моделей на текущем лучшем признаковом пространстве
        self.model_pool: Dict[str, Dict[str, Any]] = {}
        # Зафиксированные разбиения фолдов для точного согласования OOF и парных тестов
        self.fold_indices: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
        # Пофолдовые скоры последней обученной модели (для Critic Monitor)
        self.last_fold_scores: Optional[List[float]] = None

    def clear_pool(self):
        """Сброс пула моделей (вызывается при изменении признакового пространства)."""
        self.model_pool.clear()

    def prune_features(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        cat_features: List[str],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
        """Отбор признаков (Feature Pruning): удаление константных колонок и нулевого шума."""
        print("[ModelPortfolio] Запуск фильтрации неинформативных признаков (Feature Pruning)...")
        # 1. Константные колонки
        constant_cols = [c for c in X_train.columns if X_train[c].nunique() <= 1]

        # 2. Быстрая оценка важности через CatBoost на сэмпле данных
        sample_size = min(len(X_train), 6000)
        X_sample = X_train.iloc[:sample_size].copy()
        y_sample = y_train.iloc[:sample_size]

        for c in cat_features:
            if c in X_sample.columns:
                le = LabelEncoder()
                X_sample[c] = le.fit_transform(X_sample[c].astype(str))

        zero_imp_cols = []
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor

            if self.profile.task_type == "regression":
                cb = CatBoostRegressor(iterations=70, depth=4, verbose=0, thread_count=8)
                cb.fit(X_sample.fillna(-999), y_sample.values.astype(np.float32))
            else:
                le_y = LabelEncoder()
                y_enc = le_y.fit_transform(y_sample.astype(str))
                cb = CatBoostClassifier(iterations=70, depth=4, verbose=0, thread_count=8)
                cb.fit(X_sample.fillna(-999), y_enc)

            fi = cb.get_feature_importance()
            for col, imp in zip(X_sample.columns, fi):
                if imp == 0.0 and col not in cat_features:
                    zero_imp_cols.append(col)
        except Exception:
            pass

        dropped_cols = list(set(constant_cols + zero_imp_cols))

        # Защита: не отбрасывать больше 25% колонок
        max_drop = int(len(X_train.columns) * 0.25)
        if len(dropped_cols) > max_drop:
            dropped_cols = dropped_cols[:max_drop]

        if dropped_cols:
            keep_cols = [c for c in X_train.columns if c not in dropped_cols]
            print(f"[ModelPortfolio] Отсеяно {len(dropped_cols)} шумных признаков с 0 важностью: {dropped_cols[:5]}")
            X_train_pruned = X_train[keep_cols].copy()
            X_test_pruned = X_test[keep_cols].copy()
            cats_pruned = [c for c in cat_features if c in keep_cols]
            return X_train_pruned, X_test_pruned, cats_pruned, dropped_cols
        else:
            print("[ModelPortfolio] Все признаки информативны. Отсеивание не требуется.")
            return X_train, X_test, cat_features, []

    def _optimize_threshold_multipliers(
        self, oof_probs: np.ndarray, y_true: np.ndarray, verbose: bool = True
    ) -> np.ndarray:
        """Частотно-адаптивная координатная оптимизация весов классов для максимизации Macro F1.
        Редкие и ультра-редкие классы (например, Worms с 0.07% выборки) получают расширенную сетку множителей,
        а частые классы — узкую сетку для предотвращения ложноположительных срабатываний.
        """
        num_classes = oof_probs.shape[1]
        best_weights = np.ones(num_classes, dtype=np.float32)

        baseline_preds = np.argmax(oof_probs, axis=1)
        best_score = float(f1_score(y_true, baseline_preds, average="macro"))
        initial_score = best_score

        total_samples = len(y_true)
        class_counts = np.bincount(y_true, minlength=num_classes)
        class_freqs = class_counts / max(total_samples, 1)

        def get_candidates_for_class(freq: float) -> List[float]:
            if freq >= 0.15:
                # Мажорные классы (например, Normal, Generic): узкий диапазон, защита от ложных срабатываний
                return [0.5, 0.75, 0.9, 1.0, 1.15, 1.3, 1.5]
            elif freq >= 0.03:
                # Средние классы (например, Fuzzers, DoS, Reconnaissance)
                return [0.4, 0.7, 1.0, 1.4, 2.0, 3.2, 5.0]
            elif freq >= 0.005:
                # Редкие классы (например, Analysis, Backdoor, Shellcode)
                return [0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 25.0]
            else:
                # Экстремально редкие классы (Worms: < 0.1% выборки).
                # Базовая вероятность часто 0.02-0.08, поэтому множитель до 45x позволяет
                # уверенным прогнозам победить фон мажорных классов
                return [0.5, 1.0, 2.5, 6.0, 12.0, 20.0, 32.0, 48.0]

        # Оптимизируем сначала самые редкие классы, чтобы они заявили права на свои истинные примеры
        class_order = list(np.argsort(class_freqs))

        # 4 прохода координатного спуска
        for n_pass in range(4):
            improved = False
            for c in class_order:
                best_w_c = best_weights[c]
                cand_list = get_candidates_for_class(class_freqs[c])
                for mult in cand_list:
                    test_weights = best_weights.copy()
                    test_weights[c] = mult
                    scaled_probs = oof_probs * test_weights
                    preds = np.argmax(scaled_probs, axis=1)
                    score = float(f1_score(y_true, preds, average="macro"))
                    if score > best_score:
                        best_score = score
                        best_w_c = mult
                        improved = True
                best_weights[c] = best_w_c
            if not improved:
                break

        # Плотное непрерывное сканирование (17 точек) вокруг найденного оптимума для каждого класса
        for c in class_order:
            w_c = best_weights[c]
            dense_grid = np.linspace(w_c * 0.65, w_c * 1.45, 17)
            for test_w in dense_grid:
                test_weights = best_weights.copy()
                test_weights[c] = float(test_w)
                scaled_probs = oof_probs * test_weights
                preds = np.argmax(scaled_probs, axis=1)
                score = float(f1_score(y_true, preds, average="macro"))
                if score > best_score:
                    best_score = score
                    best_weights[c] = float(test_w)

        if verbose:
            print(
                f"[ModelPortfolio] Частотно-адаптивная оптимизация порогов: Macro F1 вырос с "
                f"{initial_score:.4f} до {best_score:.4f} (Δ = {best_score - initial_score:+.4f})!"
            )
        return best_weights

    def train_and_evaluate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        cat_features: List[str],
        model_family: str = "catboost",
    ) -> Tuple[float, np.ndarray, str]:
        """Обучение выбранной архитектуры («Большая тройка»: CatBoost / LightGBM / XGBoost) с 5-Fold CV."""
        X_train_enc = X_train.copy()
        X_test_enc = X_test.copy()

        # Кодирование категориальных признаков через Label Encoding
        for col in cat_features:
            if col in X_train_enc.columns:
                le = LabelEncoder()
                combined = pd.concat([X_train_enc[col], X_test_enc[col]], axis=0).astype(str)
                le.fit(combined)
                X_train_enc[col] = le.transform(X_train_enc[col].astype(str))
                X_test_enc[col] = le.transform(X_test_enc[col].astype(str))

        backend_name = model_family

        # =====================================================================
        # 1. РЕГРЕССИЯ
        # =====================================================================
        if self.profile.task_type == "regression":
            y_vals = y_train.values.astype(np.float32)
            y_min, y_max = float(np.min(y_vals)), float(np.max(y_vals))
            oof_preds = np.zeros(len(y_vals), dtype=np.float32)
            test_preds = np.zeros(len(X_test_enc), dtype=np.float32)

            # Кандидаты категориальных колонок для Out-of-Fold Target Encoding (без утечек)
            te_candidates = [
                c for c in cat_features
                if c in X_train.columns and 2 <= X_train[c].nunique() <= 120
            ][:4]

            if self.fold_indices is None or len(self.fold_indices[0][0]) + len(self.fold_indices[0][1]) != len(y_vals):
                kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
                self.fold_indices = list(kf.split(X_train_enc, y_vals))

            fold_scores = []

            for fold, (train_idx, val_idx) in enumerate(self.fold_indices):
                X_tr = X_train_enc.iloc[train_idx].copy()
                y_tr = y_vals[train_idx]
                X_va = X_train_enc.iloc[val_idx].copy()
                y_va = y_vals[val_idx]
                X_te = X_test_enc.copy()

                # Сглаженный Out-of-Fold Target Encoding
                global_y = float(np.mean(y_tr))
                for c in te_candidates:
                    raw_tr = X_train[c].iloc[train_idx]
                    raw_va = X_train[c].iloc[val_idx]
                    raw_te = X_test[c]

                    grp_stats = pd.Series(y_tr).groupby(raw_tr.values).agg(["mean", "count"])
                    smoothing = 15.0
                    smoothed_series = (grp_stats["mean"] * grp_stats["count"] + smoothing * global_y) / (grp_stats["count"] + smoothing)
                    te_map = smoothed_series.to_dict()

                    col_name = f"oof_te_{c}"
                    X_tr[col_name] = raw_tr.map(te_map).fillna(global_y).astype(np.float32)
                    X_va[col_name] = raw_va.map(te_map).fillna(global_y).astype(np.float32)
                    X_te[col_name] = raw_te.map(te_map).fillna(global_y).astype(np.float32)

                if model_family == "lightgbm":
                    try:
                        import lightgbm as lgb
                        model = lgb.LGBMRegressor(
                            n_estimators=1000,
                            learning_rate=0.04,
                            num_leaves=45,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            random_state=42 + fold,
                            verbose=-1,
                            n_jobs=8,
                        )
                        model.fit(
                            X_tr, y_tr,
                            eval_set=[(X_va, y_va)],
                            callbacks=[lgb.early_stopping(stopping_rounds=70, verbose=False)],
                        )
                        val_p = model.predict(X_va)
                        t_p = model.predict(X_te)
                    except Exception:
                        from sklearn.ensemble import HistGradientBoostingRegressor
                        model = HistGradientBoostingRegressor(max_iter=350, learning_rate=0.06, max_depth=8, random_state=42 + fold)
                        model.fit(X_tr.fillna(-999), y_tr)
                        val_p = model.predict(X_va.fillna(-999))
                        t_p = model.predict(X_te.fillna(-999))

                elif model_family == "xgboost_reg_l2":
                    # Регуляризованная модель XGBoost (L2 reg + feature subsampling 0.65)
                    try:
                        import xgboost as xgb
                        model = xgb.XGBRegressor(
                            n_estimators=1000,
                            learning_rate=0.035,
                            max_depth=7,
                            reg_lambda=5.0,
                            subsample=0.8,
                            colsample_bytree=0.65,
                            tree_method="hist",
                            random_state=42 + fold,
                            n_jobs=8,
                            early_stopping_rounds=70,
                        )
                        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                        val_p = model.predict(X_va)
                        t_p = model.predict(X_te)
                    except Exception:
                        from sklearn.ensemble import HistGradientBoostingRegressor
                        model = HistGradientBoostingRegressor(max_iter=350, learning_rate=0.06, max_depth=8, random_state=42 + fold)
                        model.fit(X_tr.fillna(-999), y_tr)
                        val_p = model.predict(X_va.fillna(-999))
                        t_p = model.predict(X_te.fillna(-999))

                elif model_family == "xgboost":
                    try:
                        import xgboost as xgb
                        model = xgb.XGBRegressor(
                            n_estimators=1000,
                            learning_rate=0.04,
                            max_depth=6,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            tree_method="hist",
                            random_state=42 + fold,
                            n_jobs=8,
                            early_stopping_rounds=70,
                        )
                        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                        val_p = model.predict(X_va)
                        t_p = model.predict(X_te)
                    except Exception:
                        from sklearn.ensemble import HistGradientBoostingRegressor
                        model = HistGradientBoostingRegressor(max_iter=350, learning_rate=0.06, max_depth=8, random_state=42 + fold)
                        model.fit(X_tr.fillna(-999), y_tr)
                        val_p = model.predict(X_va.fillna(-999))
                        t_p = model.predict(X_te.fillna(-999))

                elif model_family == "hist_gb":
                    from sklearn.ensemble import HistGradientBoostingRegressor
                    model = HistGradientBoostingRegressor(
                        max_iter=400,
                        learning_rate=0.06,
                        max_depth=8,
                        min_samples_leaf=20,
                        random_state=42 + fold,
                    )
                    model.fit(X_tr.fillna(-999), y_tr)
                    val_p = model.predict(X_va.fillna(-999))
                    t_p = model.predict(X_te.fillna(-999))

                else:  # default catboost
                    from catboost import CatBoostRegressor
                    model = CatBoostRegressor(
                        iterations=1200,
                        learning_rate=0.04,
                        depth=7,
                        thread_count=8,
                        random_seed=42 + fold,
                        verbose=0,
                    )
                    model.fit(X_tr.fillna(-999), y_tr, eval_set=(X_va.fillna(-999), y_va), early_stopping_rounds=80, verbose=False)
                    val_p = model.predict(X_va.fillna(-999))
                    t_p = model.predict(X_te.fillna(-999))

                oof_preds[val_idx] = val_p
                test_preds += t_p / self.n_splits

                # Вычисляем пофолдовый скор
                fold_rmse = float(np.sqrt(mean_squared_error(y_va, np.clip(val_p, y_min, y_max))))
                fold_scores.append(fold_rmse)

            # Физический пост-процессинг: ограничение допустимых границ таргета
            oof_preds = np.clip(oof_preds, y_min, y_max)
            test_preds = np.clip(test_preds, y_min, y_max)

            oof_score = float(np.sqrt(mean_squared_error(y_vals, oof_preds)))
            fold_scores = [
                float(np.sqrt(mean_squared_error(y_vals[va_idx], oof_preds[va_idx])))
                for tr_idx, va_idx in self.fold_indices
            ]
            self.last_fold_scores = fold_scores

            # Сохраняем в пул для последующего ансамблирования
            self.model_pool[model_family] = {
                "oof": oof_preds,
                "test": test_preds,
                "score": oof_score,
                "fold_scores": fold_scores,
                "name": model_family,
                "y_min": y_min,
                "y_max": y_max,
            }

            return oof_score, test_preds, backend_name

        # =====================================================================
        # 2. КЛАССИФИКАЦИЯ (Бинарная и Многоклассовая)
        # =====================================================================
        else:
            label_enc = LabelEncoder()
            y_encoded = label_enc.fit_transform(y_train.astype(str))
            num_classes = len(label_enc.classes_)
            oof_probs = np.zeros((len(y_encoded), num_classes), dtype=np.float32)
            test_probs = np.zeros((len(X_test_enc), num_classes), dtype=np.float32)

            # Смягченные веса классов (Square-Root Balanced) для LightGBM:
            # Предотвращают 560-кратный взрыв весов и защищают разбиения средних классов
            if num_classes > 2:
                counts = np.bincount(y_encoded, minlength=num_classes)
                raw_w = np.sqrt(len(y_encoded) / (num_classes * np.maximum(counts, 1)))
                lgb_weights = {i: float(w) for i, w in enumerate(raw_w)}
            else:
                lgb_weights = "balanced"

            # Топ-3 категориальные колонки умеренной кардинальности для OOF Target Probability Encoding
            mc_te_candidates = [
                c for c in cat_features
                if c in X_train.columns and 2 <= X_train[c].nunique() <= 150
            ][:3]

            if self.fold_indices is None or len(self.fold_indices[0][0]) + len(self.fold_indices[0][1]) != len(y_encoded):
                skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
                self.fold_indices = list(skf.split(X_train_enc, y_encoded))

            fold_scores = []

            for fold, (train_idx, val_idx) in enumerate(self.fold_indices):
                X_tr = X_train_enc.iloc[train_idx].copy()
                y_tr = y_encoded[train_idx]
                X_va = X_train_enc.iloc[val_idx].copy()
                y_va = y_encoded[val_idx]
                X_te = X_test_enc.copy()

                # Сглаженный Out-of-Fold Target Probability Encoding по классам (Class Probabilities)
                for c in mc_te_candidates:
                    raw_tr = X_train[c].iloc[train_idx]
                    raw_va = X_train[c].iloc[val_idx]
                    raw_te = X_test[c]

                    for k in range(num_classes):
                        y_k = (y_tr == k).astype(np.float32)
                        prior_k = float(np.mean(y_k))
                        grp = pd.Series(y_k).groupby(raw_tr.values).agg(["sum", "count"])
                        smoothing = 10.0
                        smoothed_p = (grp["sum"] + smoothing * prior_k) / (grp["count"] + smoothing)
                        te_map_k = smoothed_p.to_dict()

                        col_name = f"oof_te_{c}_c{k}"
                        X_tr[col_name] = raw_tr.map(te_map_k).fillna(prior_k).astype(np.float32)
                        X_va[col_name] = raw_va.map(te_map_k).fillna(prior_k).astype(np.float32)
                        X_te[col_name] = raw_te.map(te_map_k).fillna(prior_k).astype(np.float32)

                if model_family == "lightgbm":
                    try:
                        import lightgbm as lgb
                        model = lgb.LGBMClassifier(
                            n_estimators=1000,
                            learning_rate=0.04,
                            num_leaves=45,
                            class_weight=lgb_weights,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            random_state=42 + fold,
                            verbose=-1,
                            n_jobs=8,
                        )
                        model.fit(
                            X_tr, y_tr,
                            eval_set=[(X_va, y_va)],
                            callbacks=[lgb.early_stopping(stopping_rounds=70, verbose=False)],
                        )
                        val_p = model.predict_proba(X_va)
                        t_p = model.predict_proba(X_te)
                    except Exception:
                        from sklearn.ensemble import HistGradientBoostingClassifier
                        model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=8, class_weight="balanced", random_state=42 + fold)
                        model.fit(X_tr.fillna(-999), y_tr)
                        val_p = model.predict_proba(X_va.fillna(-999))
                        t_p = model.predict_proba(X_te.fillna(-999))

                elif model_family == "xgboost":
                    try:
                        import xgboost as xgb
                        model = xgb.XGBClassifier(
                            n_estimators=1000,
                            learning_rate=0.04,
                            max_depth=6,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            tree_method="hist",
                            random_state=42 + fold,
                            n_jobs=8,
                            early_stopping_rounds=70,
                        )
                        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                        val_p = model.predict_proba(X_va)
                        t_p = model.predict_proba(X_te)
                    except Exception:
                        from sklearn.ensemble import HistGradientBoostingClassifier
                        model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=8, class_weight="balanced", random_state=42 + fold)
                        model.fit(X_tr.fillna(-999), y_tr)
                        val_p = model.predict_proba(X_va.fillna(-999))
                        t_p = model.predict_proba(X_te.fillna(-999))

                elif model_family == "xgboost_deep":
                    # Глубокая архитектура XGBoost для выявления сложных сигнатур редких атак (Analysis, Backdoor)
                    try:
                        import xgboost as xgb
                        model = xgb.XGBClassifier(
                            n_estimators=1000,
                            learning_rate=0.035,
                            max_depth=8,
                            min_child_weight=2,
                            subsample=0.85,
                            colsample_bytree=0.7,
                            tree_method="hist",
                            random_state=42 + fold,
                            n_jobs=8,
                            early_stopping_rounds=70,
                        )
                        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                        val_p = model.predict_proba(X_va)
                        t_p = model.predict_proba(X_te)
                    except Exception:
                        from sklearn.ensemble import HistGradientBoostingClassifier
                        model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=8, class_weight="balanced", random_state=42 + fold)
                        model.fit(X_tr.fillna(-999), y_tr)
                        val_p = model.predict_proba(X_va.fillna(-999))
                        t_p = model.predict_proba(X_te.fillna(-999))

                elif model_family == "hist_gb":
                    from sklearn.ensemble import HistGradientBoostingClassifier
                    model = HistGradientBoostingClassifier(
                        max_iter=350,
                        learning_rate=0.06,
                        max_depth=8,
                        class_weight="balanced",
                        random_state=42 + fold,
                    )
                    model.fit(X_tr.fillna(-999), y_tr)
                    val_p = model.predict_proba(X_va.fillna(-999))
                    t_p = model.predict_proba(X_te.fillna(-999))

                else:  # default catboost
                    from catboost import CatBoostClassifier
                    cb_iters = 500 if num_classes > 2 else 900
                    cb_early = 40 if num_classes > 2 else 70
                    model = CatBoostClassifier(
                        iterations=cb_iters,
                        learning_rate=0.06,
                        depth=6,
                        loss_function="MultiClass" if num_classes > 2 else "Logloss",
                        auto_class_weights="SqrtBalanced",
                        thread_count=8,
                        random_seed=42 + fold,
                        verbose=0,
                    )
                    model.fit(X_tr.fillna(-999), y_tr, eval_set=(X_va.fillna(-999), y_va), early_stopping_rounds=cb_early, verbose=False)
                    val_p = model.predict_proba(X_va.fillna(-999))
                    t_p = model.predict_proba(X_te.fillna(-999))

                oof_probs[val_idx] = val_p
                test_probs += t_p / self.n_splits

                fold_f1 = float(f1_score(y_va, np.argmax(val_p, axis=1), average="macro"))
                fold_scores.append(fold_f1)

            # Тюнинг порогов вероятностей для редких классов
            class_weights = self._optimize_threshold_multipliers(oof_probs, y_encoded)
            scaled_oof_probs = oof_probs * class_weights
            scaled_test_probs = test_probs * class_weights

            oof_pred_labels = np.argmax(scaled_oof_probs, axis=1)
            test_pred_labels = np.argmax(scaled_test_probs, axis=1)

            final_test_preds = label_enc.inverse_transform(test_pred_labels)
            oof_score = float(f1_score(y_encoded, oof_pred_labels, average="macro"))

            fold_scores = [
                float(f1_score(y_encoded[va_idx], oof_pred_labels[va_idx], average="macro"))
                for tr_idx, va_idx in self.fold_indices
            ]
            self.last_fold_scores = fold_scores

            # Сохраняем в пул для ансамблирования
            self.model_pool[model_family] = {
                "oof_probs": oof_probs,
                "test_probs": test_probs,
                "score": oof_score,
                "fold_scores": fold_scores,
                "name": model_family,
                "label_enc": label_enc,
            }

            return oof_score, final_test_preds, backend_name

    def blend_portfolio(self, y_train: pd.Series) -> Optional[Tuple[float, np.ndarray, str]]:
        """Поиск оптимальных весов ансамбля (Blending) для моделей из текущего пула."""
        if len(self.model_pool) < 2:
            return None

        model_names = list(self.model_pool.keys())
        print(f"[ModelPortfolio] Запуск ансамблирования (Blending) моделей: {model_names}...")

        # ---------------------------------------------------------------------
        # 1. Ансамблирование для Регрессии
        # ---------------------------------------------------------------------
        if self.profile.task_type == "regression":
            y_vals = y_train.values.astype(np.float32)
            all_scores = {m: self.model_pool[m]["score"] for m in model_names}
            best_single_score = min(all_scores.values())

            # Фильтрация моделей: оставляем только те, что не более чем на 5% хуже лучшей одиночной модели
            eligible_models = [m for m in model_names if (all_scores[m] - best_single_score) / max(best_single_score, 1e-5) <= 0.05]
            if len(eligible_models) < len(model_names):
                excluded = [m for m in model_names if m not in eligible_models]
                print(f"[ModelPortfolio] Исключены отстающие модели из регрессионного блендинга: {excluded}")
            model_names = eligible_models

            if len(model_names) < 2:
                best_m = min(all_scores, key=all_scores.get)
                print(f"[ModelPortfolio] Только одна модель ({best_m}) прошла фильтр качества.")
                self.last_fold_scores = self.model_pool[best_m].get("fold_scores", None)
                return self.model_pool[best_m]["score"], self.model_pool[best_m]["test"], f"Single Best Model ({best_m})"

            y_min = self.model_pool[model_names[0]]["y_min"]
            y_max = self.model_pool[model_names[0]]["y_max"]
            oofs = [self.model_pool[m]["oof"] for m in model_names]
            tests = [self.model_pool[m]["test"] for m in model_names]

            # Лучшая одиночная модель как гарантированный нижний порог качества
            best_single_idx = int(np.argmin([self.model_pool[m]["score"] for m in model_names]))
            best_weights = np.zeros(len(model_names), dtype=np.float32)
            best_weights[best_single_idx] = 1.0
            best_rmse = self.model_pool[model_names[best_single_idx]]["score"]

            def objective(w_arr):
                pred = np.zeros_like(y_vals)
                for weight, oof in zip(w_arr, oofs):
                    pred += weight * oof
                pred = np.clip(pred, y_min, y_max)
                return float(np.sqrt(mean_squared_error(y_vals, pred)))

            # Дискретная сетка кандидатов для надежного старта
            candidates = [np.eye(len(model_names), dtype=np.float32)[i] for i in range(len(model_names))]
            candidates.append(np.full(len(model_names), 1.0 / len(model_names), dtype=np.float32))

            if len(model_names) == 2:
                candidates.extend([np.array([w, 1.0 - w], dtype=np.float32) for w in np.linspace(0.0, 1.0, 41)])
            elif len(model_names) == 3:
                for w1 in np.linspace(0.0, 1.0, 21):
                    for w2 in np.linspace(0.0, 1.0 - w1, 21):
                        w3 = max(0.0, 1.0 - w1 - w2)
                        candidates.append(np.array([w1, w2, w3], dtype=np.float32))
            else:
                for i in range(len(model_names)):
                    for j in range(i + 1, len(model_names)):
                        for w in [0.25, 0.5, 0.75]:
                            v = np.zeros(len(model_names), dtype=np.float32)
                            v[i] = w
                            v[j] = 1.0 - w
                            candidates.append(v)

            for cand_w in candidates:
                r = objective(cand_w)
                if r < best_rmse:
                    best_rmse = r
                    best_weights = cand_w

            # Непрерывная оптимизация через SLSQP
            try:
                from scipy.optimize import minimize
                bounds = [(0.0, 1.0) for _ in model_names]
                constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
                res = minimize(objective, best_weights, method="SLSQP", bounds=bounds, constraints=constraints)
                if res.success and res.fun < best_rmse:
                    best_rmse = float(res.fun)
                    best_weights = res.x / np.sum(res.x)
            except Exception:
                pass

            # Финальный взвешенный предикт линейного блендинга
            blend_test = np.zeros_like(tests[0])
            blend_oof = np.zeros_like(y_vals)
            active_components = []
            for m_name, weight, t_pred, oof in zip(model_names, best_weights, tests, oofs):
                if weight > 0.01:
                    blend_test += float(weight) * t_pred
                    blend_oof += float(weight) * oof
                    active_components.append(f"{weight:.2f}*{m_name}")
            blend_test = np.clip(blend_test, y_min, y_max)
            blend_oof = np.clip(blend_oof, y_min, y_max)
            linear_name = f"Ensemble Blend ({' + '.join(active_components)})"
            best_linear_rmse = float(np.sqrt(mean_squared_error(y_vals, blend_oof)))

            # -----------------------------------------------------------------
            # 2. Multi-Layer Stacking Meta-Learner (RidgeCV & HistGBM)
            # -----------------------------------------------------------------
            print(f"[ModelPortfolio] Оценка Stacking Meta-Learner на OOF предсказаниях...")
            S_train = np.column_stack([self.model_pool[m]["oof"] for m in model_names])
            S_test = np.column_stack([self.model_pool[m]["test"] for m in model_names])

            # Meta-Model 1: RidgeCV
            stack_oof_ridge = np.zeros(len(y_vals), dtype=np.float32)
            try:
                for tr_idx, va_idx in self.fold_indices:
                    m_reg = RidgeCV(alphas=np.logspace(-3, 3, 25))
                    m_reg.fit(S_train[tr_idx], y_vals[tr_idx])
                    stack_oof_ridge[va_idx] = m_reg.predict(S_train[va_idx])
                stack_oof_ridge = np.clip(stack_oof_ridge, y_min, y_max)
                stack_rmse_ridge = float(np.sqrt(mean_squared_error(y_vals, stack_oof_ridge)))

                meta_ridge_full = RidgeCV(alphas=np.logspace(-3, 3, 25))
                meta_ridge_full.fit(S_train, y_vals)
                stack_test_ridge = np.clip(meta_ridge_full.predict(S_test), y_min, y_max)
            except Exception as e:
                stack_rmse_ridge = 999.0
                stack_test_ridge = blend_test

            # Meta-Model 2: Shallow HistGBM
            stack_oof_hgb = np.zeros(len(y_vals), dtype=np.float32)
            try:
                for fold, (tr_idx, va_idx) in enumerate(self.fold_indices):
                    m_hgb = HistGradientBoostingRegressor(
                        max_depth=3, max_iter=60, learning_rate=0.04, l2_regularization=2.0, random_state=42 + fold
                    )
                    m_hgb.fit(S_train[tr_idx], y_vals[tr_idx])
                    stack_oof_hgb[va_idx] = m_hgb.predict(S_train[va_idx])
                stack_oof_hgb = np.clip(stack_oof_hgb, y_min, y_max)
                stack_rmse_hgb = float(np.sqrt(mean_squared_error(y_vals, stack_oof_hgb)))

                meta_hgb_full = HistGradientBoostingRegressor(
                    max_depth=3, max_iter=60, learning_rate=0.04, l2_regularization=2.0, random_state=42
                )
                meta_hgb_full.fit(S_train, y_vals)
                stack_test_hgb = np.clip(meta_hgb_full.predict(S_test), y_min, y_max)
            except Exception as e:
                stack_rmse_hgb = 999.0
                stack_test_hgb = blend_test

            # Выбор сильнейшей стратегии ансамблирования
            options = [
                (best_linear_rmse, blend_test, blend_oof, linear_name),
                (stack_rmse_ridge, stack_test_ridge, stack_oof_ridge, f"Stacking Meta-Learner (RidgeCV: {' + '.join(model_names)})"),
                (stack_rmse_hgb, stack_test_hgb, stack_oof_hgb, f"Stacking Meta-Learner (HistGBM: {' + '.join(model_names)})"),
            ]
            options.sort(key=lambda x: x[0])
            winner_score, winner_test, winner_oof, winner_name = options[0]

            winner_fold_scores = [
                float(np.sqrt(mean_squared_error(y_vals[va_idx], winner_oof[va_idx])))
                for tr_idx, va_idx in self.fold_indices
            ]
            self.last_fold_scores = winner_fold_scores

            print(f"[ModelPortfolio] Оптимальный ансамбль: {winner_name} -> CV RMSE: {winner_score:.4f} "
                  f"(Linear: {best_linear_rmse:.4f}, RidgeCV: {stack_rmse_ridge:.4f}, HistGBM: {stack_rmse_hgb:.4f})")
            return winner_score, winner_test, winner_name

        # ---------------------------------------------------------------------
        # 2. Ансамблирование для Классификации
        # ---------------------------------------------------------------------
        else:
            all_scores = {m: self.model_pool[m]["score"] for m in model_names}
            best_single_score = max(all_scores.values())

            # Фильтруем модели: исключаем модели, отстающие более чем на 1.5% по Macro F1
            eligible_models = [m for m in model_names if (best_single_score - all_scores[m]) <= 0.015]
            if len(eligible_models) < len(model_names):
                excluded = [m for m in model_names if m not in eligible_models]
                print(f"[ModelPortfolio] Исключены отстающие модели из классификационного блендинга: {excluded}")
            model_names = eligible_models

            label_enc = self.model_pool[model_names[0]]["label_enc"]
            y_encoded = label_enc.transform(y_train.astype(str))

            if len(model_names) < 2:
                best_m = max(all_scores, key=all_scores.get)
                print(f"[ModelPortfolio] Только одна модель ({best_m}) прошла фильтр качества.")
                m_info = self.model_pool[best_m]
                w = self._optimize_threshold_multipliers(m_info["oof_probs"], y_encoded, verbose=False)
                sc_test = m_info["test_probs"] * w
                preds = label_enc.inverse_transform(np.argmax(sc_test, axis=1))
                self.last_fold_scores = m_info.get("fold_scores", None)
                return m_info["score"], preds, f"Single Best Model ({best_m})"

            p_oofs = [self.model_pool[m]["oof_probs"] for m in model_names]
            p_tests = [self.model_pool[m]["test_probs"] for m in model_names]

            best_single_idx = int(np.argmax([self.model_pool[m]["score"] for m in model_names]))
            best_weights = np.zeros(len(model_names), dtype=np.float32)
            best_weights[best_single_idx] = 1.0
            best_f1 = self.model_pool[model_names[best_single_idx]]["score"]
            best_scaled_test = None

            if len(model_names) == 2:
                candidates = [np.array([w, 1.0 - w], dtype=np.float32) for w in np.linspace(0.0, 1.0, 41)]
            elif len(model_names) == 3:
                candidates = []
                for w1 in np.linspace(0.0, 1.0, 21):
                    for w2 in np.linspace(0.0, 1.0 - w1, 21):
                        w3 = max(0.0, 1.0 - w1 - w2)
                        candidates.append(np.array([w1, w2, w3], dtype=np.float32))
            else:
                candidates = [np.eye(len(model_names), dtype=np.float32)[i] for i in range(len(model_names))]
                candidates.append(np.full(len(model_names), 1.0 / len(model_names), dtype=np.float32))
                for i in range(len(model_names)):
                    for j in range(i + 1, len(model_names)):
                        for w in [0.2, 0.35, 0.5, 0.65, 0.8]:
                            v = np.zeros(len(model_names), dtype=np.float32)
                            v[i] = w
                            v[j] = 1.0 - w
                            candidates.append(v)
                for w1 in [0.35, 0.5, 0.65]:
                    for w2 in [0.15, 0.25, 0.35]:
                        for w3 in [0.05, 0.15]:
                            rem = 1.0 - w1 - w2 - w3
                            if rem >= 0:
                                candidates.append(np.array([w1, w2, w3, rem], dtype=np.float32))

            best_preds_linear_oof = None
            for cand_w in candidates:
                blend_p_oof = np.zeros_like(p_oofs[0])
                blend_p_test = np.zeros_like(p_tests[0])
                for w, oof, test in zip(cand_w, p_oofs, p_tests):
                    blend_p_oof += w * oof
                    blend_p_test += w * test

                weights = self._optimize_threshold_multipliers(blend_p_oof, y_encoded, verbose=False)
                scaled_oof = blend_p_oof * weights
                preds_oof = np.argmax(scaled_oof, axis=1)
                score = float(f1_score(y_encoded, preds_oof, average="macro"))

                if score > best_f1:
                    best_f1 = score
                    best_weights = cand_w
                    best_scaled_test = blend_p_test * weights
                    best_preds_linear_oof = preds_oof

            if best_preds_linear_oof is None:
                m_single = self.model_pool[model_names[best_single_idx]]
                w_s = self._optimize_threshold_multipliers(m_single["oof_probs"], y_encoded, verbose=False)
                best_preds_linear_oof = np.argmax(m_single["oof_probs"] * w_s, axis=1)

            active_components = []
            final_p_test = np.zeros_like(p_tests[0])
            for m_name, weight, t_probs in zip(model_names, best_weights, p_tests):
                if weight > 0.01:
                    final_p_test += float(weight) * t_probs
                    active_components.append(f"{weight:.2f}*{m_name}")

            if best_scaled_test is not None:
                final_test_preds = label_enc.inverse_transform(np.argmax(best_scaled_test, axis=1))
            else:
                final_test_preds = label_enc.inverse_transform(np.argmax(final_p_test, axis=1))

            linear_name = f"Ensemble Blend ({' + '.join(active_components)})"
            best_linear_f1 = best_f1

            # -----------------------------------------------------------------
            # 2. Multi-Layer Stacking Meta-Learner (LogisticRegression)
            # -----------------------------------------------------------------
            print(f"[ModelPortfolio] Оценка Stacking Meta-Learner на OOF вероятностях...")
            stack_f1 = -1.0
            stack_test_preds = None
            preds_stack_oof = None
            try:
                num_classes = len(label_enc.classes_)
                S_train = np.hstack([self.model_pool[m]["oof_probs"] for m in model_names])
                S_test = np.hstack([self.model_pool[m]["test_probs"] for m in model_names])

                stack_oof_probs = np.zeros((len(y_encoded), num_classes), dtype=np.float32)
                for tr_idx, va_idx in self.fold_indices:
                    lr = LogisticRegression(C=0.5, max_iter=300, solver="lbfgs")
                    lr.fit(S_train[tr_idx], y_encoded[tr_idx])
                    stack_oof_probs[va_idx] = lr.predict_proba(S_train[va_idx])

                w_stack = self._optimize_threshold_multipliers(stack_oof_probs, y_encoded, verbose=False)
                scaled_stack_oof = stack_oof_probs * w_stack
                preds_stack_oof = np.argmax(scaled_stack_oof, axis=1)
                stack_f1 = float(f1_score(y_encoded, preds_stack_oof, average="macro"))

                lr_full = LogisticRegression(C=0.5, max_iter=300, solver="lbfgs")
                lr_full.fit(S_train, y_encoded)
                stack_test_probs = lr_full.predict_proba(S_test)
                scaled_stack_test = stack_test_probs * w_stack
                stack_test_preds = label_enc.inverse_transform(np.argmax(scaled_stack_test, axis=1))
            except Exception as e:
                print(f"[ModelPortfolio] Исключение в Stacking: {e}")

            if stack_f1 > best_linear_f1:
                winner_score = stack_f1
                winner_test = stack_test_preds
                winner_oof_preds = preds_stack_oof
                winner_name = f"Stacking Meta-Learner (LogisticRegression: {' + '.join(model_names)})"
            else:
                winner_score = best_linear_f1
                winner_test = final_test_preds
                winner_oof_preds = best_preds_linear_oof
                winner_name = linear_name

            winner_fold_scores = [
                float(f1_score(y_encoded[va_idx], winner_oof_preds[va_idx], average="macro"))
                for tr_idx, va_idx in self.fold_indices
            ]
            self.last_fold_scores = winner_fold_scores

            print(f"[ModelPortfolio] Оптимальный ансамбль: {winner_name} -> CV Macro F1: {winner_score:.4f} "
                  f"(Linear: {best_linear_f1:.4f}, Stacking: {stack_f1:.4f})")
            return winner_score, winner_test, winner_name
