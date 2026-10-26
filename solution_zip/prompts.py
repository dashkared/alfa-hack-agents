"""Typed experiment passport (`TaskProfile`) and LLM prompt templates."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TaskProfile:
    """Structured experiment passport (see PROMPT.md §4.2)."""

    task_type: str = "regression"                 # regression | binary | multiclass | multilabel
    target_column: str = "target"
    id_column: str = "id"
    metric: str = "RMSE"
    objective: str = "minimize"                   # minimize | maximize
    description_summary: str = ""
    # structure of the dataset
    time_column: Optional[str] = None
    time_periods: List[Any] = field(default_factory=list)
    test_periods: List[Any] = field(default_factory=list)
    temporal_extrapolation: bool = False
    entity_columns: List[str] = field(default_factory=list)
    entity_score: float = 0.0
    entity_coverage: float = 0.0
    # classification details
    class_labels: List[str] = field(default_factory=list)
    n_classes: int = 0
    predict_proba: bool = False                   # submit probabilities instead of labels
    target_columns: List[str] = field(default_factory=list)   # multilabel case
    # numeric target range for physical clipping
    target_min: Optional[float] = None
    target_max: Optional[float] = None
    target_transform: str = "none"                # none | log1p
    # bookkeeping
    n_train: int = 0
    n_test: int = 0
    notes: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    @property
    def is_classification(self) -> bool:
        return self.task_type in ("binary", "multiclass", "multilabel")

    @property
    def is_regression(self) -> bool:
        return self.task_type == "regression"

    def pretty(self) -> str:
        lines = [
            f"тип задачи        : {self.task_type}",
            f"целевой признак   : {self.target_column}",
            f"id колонка        : {self.id_column}",
            f"метрика           : {self.metric} ({self.objective})",
            f"временная колонка : {self.time_column or '—'}",
            f"периоды train     : {self.time_periods[:8]}",
            f"периоды test      : {self.test_periods[:8]}",
            f"сущность (entity) : {self.entity_columns or '—'} (score={self.entity_score:.3f}, "
            f"coverage={self.entity_coverage:.2f})",
        ]
        if self.is_classification:
            lines.append(f"классов           : {self.n_classes}")
        return "\n".join(f"      • {l}" for l in lines)

    def router_report(self) -> Dict[str, Any]:
        """Structured router passport (spec §5.2): task type, metric, model portfolio."""
        families = []
        if self.is_regression:
            families = [
                {"name": "catboost", "suitability_score": 0.95,
                 "rationale": "oblivious boosting — силён на смешанных табличных данных"},
                {"name": "lightgbm", "suitability_score": 0.92,
                 "rationale": "листовой бустинг, быстрый на широких таблицах"},
                {"name": "xgboost", "suitability_score": 0.88,
                 "rationale": "регуляризованный бустинг для разнообразия ансамбля"},
                {"name": "hist_gb", "suitability_score": 0.85,
                 "rationale": "гистограммный бустинг sklearn как независимая реализация"},
                {"name": "extra_trees", "suitability_score": 0.75,
                 "rationale": "рандомизированный бэггинг снижает дисперсию"},
                {"name": "ridge", "suitability_score": 0.65,
                 "rationale": "линейная регуляризованная база для блендинга"},
            ]
        elif self.is_classification:
            families = [
                {"name": "lightgbm", "suitability_score": 0.93,
                 "rationale": "мультиклассовый бустинг с нативной поддержкой категорий"},
                {"name": "catboost", "suitability_score": 0.91,
                 "rationale": "устойчив к дисбалансу классов, oblivious-деревья"},
                {"name": "xgboost", "suitability_score": 0.87,
                 "rationale": "softprob-бустинг с регуляризацией"},
                {"name": "hist_gb", "suitability_score": 0.84,
                 "rationale": "быстрый гистограммный бустинг"},
                {"name": "logreg", "suitability_score": 0.66,
                 "rationale": "линейная модель для калибровки вероятностей"},
            ]
        return {
            "task_type": self.task_type,
            "target": self.target_column,
            "input_modality": ["tabular"] + (["time_series"] if self.time_column else []),
            "supervision": "supervised",
            "model_families": families,
            "constraints": {"latency_sensitive": False, "max_training_time_sec": 9000,
                            "memory_limit_mb": 32768},
            "recommended_metric": self.metric,
            "ambiguity_level": "low" if self.metric else "medium",
            "detected_metric": self.metric,
            "temporal_extrapolation": bool(self.temporal_extrapolation),
            "entity_key_candidates": self.entity_columns,
            "n_train": self.n_train,
            "n_test": self.n_test,
        }


# --------------------------------------------------------------------------- #
# LLM prompts (used sparingly: the platform LLM is a slow reasoning model)
# --------------------------------------------------------------------------- #
PROFILER_SYSTEM_PROMPT = """Ты — агент-роутер AutoML. По описанию задачи и словарю признаков верни СТРОГО JSON:
{
  "task_type": "regression|binary|multiclass",
  "target": "имя целевой колонки",
  "metric": "RMSE|MAE|R2|Macro F1|ROC-AUC|Accuracy|LogLoss",
  "objective": "minimize|maximize",
  "time_column": "имя колонки периода или null",
  "entity_columns": ["колонки, идентифицирующие один и тот же объект между строками"],
  "submission_kind": "labels|probabilities",
  "ambiguity": "low|medium|high",
  "rationale": "одно предложение"
}
Только JSON, без markdown."""

FEATURE_SYSTEM_PROMPT = """Ты — инженер признаков. По описанию задачи и списку колонок предложи ОДНУ
содержательную формулу новых признаков. Верни СТРОГО JSON:
{
  "hypothesis": "краткое название гипотезы",
  "rationale": "зачем это должно помочь",
  "python_code": "def add_features(df):\\n    ...\\n    return df"
}
Требования к коду:
- функция `add_features(df)` принимает pandas DataFrame и возвращает DataFrame с теми же строками;
- можно использовать numpy (np) и pandas (pd), они уже импортированы;
- НЕЛЬЗЯ: читать файлы, сеть, subprocess, print, изменения исходных колонок;
- добавляй не более 4 новых колонок, защищайся от деления на ноль (+1e-5) и логирования отрицательных (np.log1p(x.clip(lower=0)));
- код должен быть детерминированным и работать без исключений.
Только JSON, без markdown."""

CRITIC_SYSTEM_PROMPT = """Ты — критик ML-экспериментов. По истории экспериментов реши: продолжать поиск
признаков или остановиться. Верни СТРОГО JSON:
{"decision": "continue|stop", "what_worked": "...", "next_direction": "..."}"""


def profiler_user_prompt(task_text: str, dictionary_text: str, data_overview: str) -> str:
    return f"""=== ОПИСАНИЕ ЗАДАЧИ ===
{task_text[:3500]}

=== СЛОВАРЬ ПРИЗНАКОВ (фрагмент) ===
{dictionary_text[:1500]}

=== СХЕМА ДАННЫХ ===
{data_overview[:1500]}
"""


def feature_user_prompt(task_text: str, columns_text: str, hypothesis_history: str) -> str:
    return f"""=== ОПИСАНИЕ ЗАДАЧИ ===
{task_text[:2200]}

=== ДОСТУПНЫЕ КОЛОНКИ ===
{columns_text[:2500]}

=== УЖЕ ИСПРОБОВАННЫЕ ГИПОТЕЗЫ ===
{hypothesis_history[:1500]}
"""
