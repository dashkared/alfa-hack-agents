import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from prompts import TaskProfile
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=UserWarning)


class ModelPortfolioAgent:
    """Агент 3: ML-Архитектор.

    Управляет портфелем моделей «Большой тройки» (CatBoost, LightGBM, XGBoost),
    выполняет очистку неинформативных признаков (Feature Pruning),
    оптимизирует многомерные пороги для редких классов и строит оптимальные ансамбли (OOF Blending).
    """

    def __init__(self, profile: TaskProfile, n_splits: int = 5):
        self.profile = profile
        self.n_splits = n_splits
        # Пул обученных моделей на текущем лучшем признаковом пространстве
        self.model_pool: Dict[str, Dict[str, Any]] = {}

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
        """Многомерная координатная оптимизация весов классов для максимизации Macro F1."""
        num_classes = oof_probs.shape[1]
        best_weights = np.ones(num_classes, dtype=np.float32)

        baseline_preds = np.argmax(oof_probs, axis=1)
        best_score = float(f1_score(y_true, baseline_preds, average="macro"))
        initial_score = best_score

        # Геометрическая сетка Precision-Guard: предотвращает лавинообразные ложные срабатывания (False Positives)
        candidates = [0.25, 0.45, 0.65, 0.85, 1.0, 1.25, 1.6, 2.0, 2.8, 3.8]

        # 3 прохода координатного спуска
        for n_pass in range(3):
            improved = False
            for c in range(num_classes):
                best_w_c = best_weights[c]
                for mult in candidates:
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

        if verbose:
            print(
                f"[ModelPortfolio] Оптимизация порогов: Macro F1 вырос с "
                f"{initial_score:.4f} до {best_score:.4f}!"
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

            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

            for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_enc, y_vals)):
                X_tr, y_tr = X_train_enc.iloc[train_idx], y_vals[train_idx]
                X_va, y_va = X_train_enc.iloc[val_idx], y_vals[val_idx]

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
                        t_p = model.predict(X_test_enc)
                    except Exception:
                        from sklearn.ensemble import HistGradientBoostingRegressor
                        model = HistGradientBoostingRegressor(max_iter=350, learning_rate=0.06, max_depth=8, random_state=42 + fold)
                        model.fit(X_tr.fillna(-999), y_tr)
                        val_p = model.predict(X_va.fillna(-999))
                        t_p = model.predict(X_test_enc.fillna(-999))

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
                        t_p = model.predict(X_test_enc)
                    except Exception:
                        from sklearn.ensemble import HistGradientBoostingRegressor
                        model = HistGradientBoostingRegressor(max_iter=350, learning_rate=0.06, max_depth=8, random_state=42 + fold)
                        model.fit(X_tr.fillna(-999), y_tr)
                        val_p = model.predict(X_va.fillna(-999))
                        t_p = model.predict(X_test_enc.fillna(-999))

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
                    t_p = model.predict(X_test_enc.fillna(-999))

                else:  # default catboost
                    from catboost import CatBoostRegressor
                    model = CatBoostRegressor(
                        iterations=1000,
                        learning_rate=0.05,
                        depth=6,
                        thread_count=8,
                        random_seed=42 + fold,
                        verbose=0,
                    )
                    model.fit(X_tr.fillna(-999), y_tr, eval_set=(X_va.fillna(-999), y_va), early_stopping_rounds=70, verbose=False)
                    val_p = model.predict(X_va.fillna(-999))
                    t_p = model.predict(X_test_enc.fillna(-999))

                oof_preds[val_idx] = val_p
                test_preds += t_p / self.n_splits

            # Физический пост-процессинг: ограничение допустимых границ таргета
            oof_preds = np.clip(oof_preds, y_min, y_max)
            test_preds = np.clip(test_preds, y_min, y_max)

            oof_score = float(np.sqrt(mean_squared_error(y_vals, oof_preds)))

            # Сохраняем в пул для последующего ансамблирования
            self.model_pool[model_family] = {
                "oof": oof_preds,
                "test": test_preds,
                "score": oof_score,
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

            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)

            for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_enc, y_encoded)):
                X_tr, y_tr = X_train_enc.iloc[train_idx], y_encoded[train_idx]
                X_va, y_va = X_train_enc.iloc[val_idx], y_encoded[val_idx]

                if model_family == "lightgbm":
                    try:
                        import lightgbm as lgb
                        model = lgb.LGBMClassifier(
                            n_estimators=1000,
                            learning_rate=0.04,
                            num_leaves=45,
                            class_weight="balanced",
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
                        t_p = model.predict_proba(X_test_enc)
                    except Exception:
                        from sklearn.ensemble import HistGradientBoostingClassifier
                        model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=8, class_weight="balanced", random_state=42 + fold)
                        model.fit(X_tr.fillna(-999), y_tr)
                        val_p = model.predict_proba(X_va.fillna(-999))
                        t_p = model.predict_proba(X_test_enc.fillna(-999))

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
                        t_p = model.predict_proba(X_test_enc)
                    except Exception:
                        from sklearn.ensemble import HistGradientBoostingClassifier
                        model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=8, class_weight="balanced", random_state=42 + fold)
                        model.fit(X_tr.fillna(-999), y_tr)
                        val_p = model.predict_proba(X_va.fillna(-999))
                        t_p = model.predict_proba(X_test_enc.fillna(-999))

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
                    t_p = model.predict_proba(X_test_enc.fillna(-999))

                else:  # default catboost
                    from catboost import CatBoostClassifier
                    cb_iters = 500 if num_classes > 2 else 900
                    cb_early = 40 if num_classes > 2 else 70
                    model = CatBoostClassifier(
                        iterations=cb_iters,
                        learning_rate=0.06,
                        depth=6,
                        loss_function="MultiClass" if num_classes > 2 else "Logloss",
                        auto_class_weights="Balanced",
                        thread_count=8,
                        random_seed=42 + fold,
                        verbose=0,
                    )
                    model.fit(X_tr.fillna(-999), y_tr, eval_set=(X_va.fillna(-999), y_va), early_stopping_rounds=cb_early, verbose=False)
                    val_p = model.predict_proba(X_va.fillna(-999))
                    t_p = model.predict_proba(X_test_enc.fillna(-999))

                oof_probs[val_idx] = val_p
                test_probs += t_p / self.n_splits

            # Тюнинг порогов вероятностей для редких классов
            class_weights = self._optimize_threshold_multipliers(oof_probs, y_encoded)
            scaled_oof_probs = oof_probs * class_weights
            scaled_test_probs = test_probs * class_weights

            oof_pred_labels = np.argmax(scaled_oof_probs, axis=1)
            test_pred_labels = np.argmax(scaled_test_probs, axis=1)

            final_test_preds = label_enc.inverse_transform(test_pred_labels)
            oof_score = float(f1_score(y_encoded, oof_pred_labels, average="macro"))

            # Сохраняем в пул для ансамблирования
            self.model_pool[model_family] = {
                "oof_probs": oof_probs,
                "test_probs": test_probs,
                "score": oof_score,
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
            candidates = []
            if len(model_names) == 2:
                candidates = [np.array([w, 1.0 - w], dtype=np.float32) for w in np.linspace(0.0, 1.0, 21)]
            elif len(model_names) == 3:
                for w1 in np.linspace(0.0, 1.0, 15):
                    for w2 in np.linspace(0.0, 1.0 - w1, 15):
                        w3 = max(0.0, 1.0 - w1 - w2)
                        candidates.append(np.array([w1, w2, w3], dtype=np.float32))

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

            # Финальный взвешенный предикт
            blend_test = np.zeros_like(tests[0])
            active_components = []
            for m_name, weight, t_pred in zip(model_names, best_weights, tests):
                if weight > 0.01:
                    blend_test += float(weight) * t_pred
                    active_components.append(f"{weight:.2f}*{m_name}")
            blend_test = np.clip(blend_test, y_min, y_max)

            name = f"Ensemble Blend ({' + '.join(active_components)})"
            print(f"[ModelPortfolio] Оптимальный ансамбль: {name} -> CV RMSE: {best_rmse:.4f}")
            return best_rmse, blend_test, name

        # ---------------------------------------------------------------------
        # 2. Ансамблирование для Классификации
        # ---------------------------------------------------------------------
        else:
            label_enc = self.model_pool[model_names[0]]["label_enc"]
            y_encoded = label_enc.transform(y_train.astype(str))
            p_oofs = [self.model_pool[m]["oof_probs"] for m in model_names]
            p_tests = [self.model_pool[m]["test_probs"] for m in model_names]

            best_single_idx = int(np.argmax([self.model_pool[m]["score"] for m in model_names]))
            best_weights = np.zeros(len(model_names), dtype=np.float32)
            best_weights[best_single_idx] = 1.0
            best_f1 = self.model_pool[model_names[best_single_idx]]["score"]
            best_scaled_test = None

            if len(model_names) == 2:
                candidates = [np.array([w, 1.0 - w], dtype=np.float32) for w in np.linspace(0.0, 1.0, 21)]
            else:
                candidates = []
                for w1 in np.linspace(0.0, 1.0, 15):
                    for w2 in np.linspace(0.0, 1.0 - w1, 15):
                        w3 = max(0.0, 1.0 - w1 - w2)
                        candidates.append(np.array([w1, w2, w3], dtype=np.float32))

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

            name = f"Ensemble Blend ({' + '.join(active_components)})"
            print(f"[ModelPortfolio] Оптимальный ансамбль: {name} -> CV Macro F1: {best_f1:.4f}")
            return best_f1, final_test_preds, name
