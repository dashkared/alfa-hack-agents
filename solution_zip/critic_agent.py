from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from data_utils import save_atomic_submission
from prompts import CriticVerdict, TaskProfile


class CriticMonitorAgent:
    """Агент 4: Монитор качества и Судья (по PIPELINE.md §8).
    Оценивает результаты, ставит вердикты PASS/FAIL и управляет критерием останова.
    Включает парный статистический контроль по фолдам кросс-валидации.
    """

    def __init__(self, task_dir: Path, profile: TaskProfile, patience: int = 2):
        self.task_dir = Path(task_dir)
        self.profile = profile
        self.best_score: Optional[float] = None
        self.best_iteration: int = -1
        self.best_fold_scores: Optional[List[float]] = None
        self.patience = patience
        self.consecutive_fails: int = 0
        self.history = []

    def should_stop(self) -> Tuple[bool, str]:
        """Критерий останова: проверка на стагнацию (patience)."""
        if self.consecutive_fails >= self.patience:
            return True, f"Сработал критерий останова: {self.consecutive_fails} неудачных гипотез подряд без улучшения (Stagnation Early Stopping)."
        return False, ""

    def evaluate_experiment(
        self,
        iteration: int,
        hypothesis_name: str,
        current_cv_score: float,
        test_predictions: np.ndarray,
        test_ids: pd.Series,
        fold_scores: Optional[List[float]] = None,
    ) -> CriticVerdict:
        """Оценка итогов эксперимента, вынесение вердикта и чекпоинт лучшего сабмита.
        Использует парный статистический контроль по фолдам кросс-валидации (PIPELINE.md §8).
        """
        is_minimize = (self.profile.objective == "minimize")

        # ---------------------------------------------------------------------
        # Итерация 0: Фиксация начальной точки отсчета
        # ---------------------------------------------------------------------
        if self.best_score is None:
            self.best_score = current_cv_score
            self.best_iteration = iteration
            self.best_fold_scores = fold_scores.copy() if fold_scores else None
            self.consecutive_fails = 0
            save_atomic_submission(self.task_dir, test_ids, test_predictions, self.profile)

            verdict = CriticVerdict(
                status="PASS",
                current_cv_score=current_cv_score,
                previous_best_score=current_cv_score,
                delta=0.0,
                is_improvement=True,
                critique_for_agents="Базовый бейзлайн зафиксирован как точка отсчета. Начинаем итеративное улучшение.",
            )
            self.history.append((iteration, hypothesis_name, current_cv_score, "PASS", 0.0))
            return verdict

        # ---------------------------------------------------------------------
        # Оценка направления улучшения метрики
        # ---------------------------------------------------------------------
        if is_minimize:
            delta = self.best_score - current_cv_score
            is_better = current_cv_score < self.best_score
        else:
            delta = current_cv_score - self.best_score
            is_better = current_cv_score > self.best_score

        # ---------------------------------------------------------------------
        # Парный статистический контроль по фолдам (по PIPELINE.md §8)
        # ---------------------------------------------------------------------
        fold_support_ratio = 1.0
        fold_msg = ""
        if fold_scores is not None and self.best_fold_scores is not None and len(fold_scores) == len(self.best_fold_scores):
            fold_deltas = [
                (bf - cf) if is_minimize else (cf - bf)
                for cf, bf in zip(fold_scores, self.best_fold_scores)
            ]
            improved_folds = sum(1 for d in fold_deltas if d > 0)
            total_folds = len(fold_deltas)
            fold_support_ratio = improved_folds / total_folds
            fold_msg = f" (улучшение на {improved_folds}/{total_folds} фолдах)"

        # ---------------------------------------------------------------------
        # Логика вердикта по PIPELINE.md §8
        # ---------------------------------------------------------------------
        if is_better and delta > 0.0:
            # Защита от оверфита на случайный одиночный фолд:
            # Если общий скор формально вырос, но улучшился строго 1 фолд из 5 (а на 4 фолдах деградация),
            # классифицируем как UNCERTAIN для предотвращения зашумления сабмита.
            if fold_support_ratio < 0.35 and len(fold_scores or []) >= 4:
                status = "UNCERTAIN"
                self.consecutive_fails += 1
                critique = (
                    f"Дельта положительна ({delta:+.4f}), но вызвана случайным выбросом одного фолда{fold_msg}. "
                    "Откат для защиты от переобучения на валидацию."
                )
            else:
                status = "PASS"
                self.consecutive_fails = 0
                self.best_score = current_cv_score
                self.best_iteration = iteration
                if fold_scores:
                    self.best_fold_scores = fold_scores.copy()

                # Чекпоинт: сохраняем улучшенный сабмит
                save_atomic_submission(self.task_dir, test_ids, test_predictions, self.profile)
                critique = (
                    f"Отличная гипотеза! Метрика улучшилась на {abs(delta):.4f}{fold_msg}. "
                    "Эти признаки приняты в основу. Развивай это направление дальше."
                )
        elif abs(delta) < 0.0005:
            # Дельта около нуля без статистического улучшения
            status = "UNCERTAIN"
            self.consecutive_fails += 1
            critique = (
                f"Изменение в пределах шума (дельта {delta:+.5f}){fold_msg}. "
                "Гипотеза не дает статистического преимущества. Попробуй более контрастные признаки."
            )
        else:
            status = "FAIL"
            self.consecutive_fails += 1
            critique = (
                f"Метрика ухудшилась на {abs(delta):.4f}{fold_msg} (признаки вызвали переобучение или зашумление). "
                "ОТКАТ к предыдущему стабильному состоянию. Смени подход: попробуй нелинейные соотношения или агрегаты."
            )

        verdict = CriticVerdict(
            status=status,
            current_cv_score=current_cv_score,
            previous_best_score=self.best_score,
            delta=float(delta),
            is_improvement=is_better,
            critique_for_agents=critique,
        )

        self.history.append((iteration, hypothesis_name, current_cv_score, status, float(delta)))
        return verdict
