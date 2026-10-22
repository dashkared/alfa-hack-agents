from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, model_validator


# =============================================================================
# 1. СХЕМА АГЕНТА 1: TaskProfiler (Паспорт эксперимента по PIPELINE.md §1.1)
# =============================================================================

class TaskProfile(BaseModel):
    """Структурированный паспорт эксперимента и задачи машинного обучения."""
    task_type: Literal["regression", "binary_classification", "multiclass_classification"] = Field(
        description="Тип ML-задачи: regression, binary_classification или multiclass_classification"
    )
    target_column: str = Field(
        description="Название целевой колонки (target) в обучающей выборке"
    )
    id_column: str = Field(
        default="id",
        description="Название колонки с идентификатором строки"
    )
    metric: str = Field(
        description="Основная целевая метрика оптимизации (RMSE, Macro F1, ROC-AUC, Accuracy)"
    )
    objective: Literal["minimize", "maximize"] = Field(
        default="minimize",
        description="Направление оптимизации: minimize (RMSE/MAE) или maximize (Macro F1/ROC-AUC)"
    )
    modality: str = Field(
        default="tabular",
        description="Модальность входных данных (tabular)"
    )
    description_summary: Optional[str] = Field(
        default="",
        description="Краткая формулировка физического/бизнес-смысла задачи"
    )


PROFILER_SYSTEM_PROMPT = """Ты — ведущий эксперт по машинному обучению и Data Science (Agent 1: TaskProfiler).
Твоя задача — внимательно изучить условие ML-задачи (task.md) и структуру колонок датасета, после чего точно определить:
1. task_type: "regression", "binary_classification" или "multiclass_classification".
2. target_column: точное имя колонки, которую нужно предсказать.
3. id_column: колонка с идентификатором строки (обычно "id").
4. metric: целевая метрика (например, "RMSE" или "Macro F1").
5. objective: "minimize" (для ошибок вроде RMSE/MSE) или "maximize" (для F1, точности, ROC-AUC).

Ответь СТРОГО валидным JSON-объектом в формате схемы TaskProfile без лишних рассуждений и текста.
"""


# =============================================================================
# 2. СХЕМЫ АГЕНТА 2: FeatureArchitect (Инженер признаков по PIPELINE.md §2.4)
# =============================================================================

class FeatureHypothesis(BaseModel):
    """Гипотеза создания отдельного признака или группы признаков."""
    feature_name: Optional[str] = Field(default="feat", description="Имя нового признака")
    rationale: Optional[str] = Field(default="", description="Физический/доменный смысл признака")
    formula_expression: Optional[str] = Field(default="", description="Формула признака")

    @model_validator(mode="before")
    @classmethod
    def normalize_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            name = data.get("feature_name") or data.get("name") or "feat"
            rationale = data.get("rationale") or data.get("description") or ""
            formula = data.get("formula_expression") or data.get("formula") or ""
            return {"feature_name": str(name), "rationale": str(rationale), "formula_expression": str(formula)}
        return data


class FeatureProposal(BaseModel):
    """Пакет гипотез и исполняемый Python-код для генерации признаков с поддержкой алиасов."""
    hypotheses: List[FeatureHypothesis] = Field(default_factory=list)
    python_code: str = Field(default="")

    @model_validator(mode="before")
    @classmethod
    def normalize_input(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Поддержка альтернативных ключей от LLM
            hyps = data.get("hypotheses") or data.get("features") or data.get("hypotheses_list") or []
            code = data.get("python_code") or data.get("code") or data.get("function_code") or ""
            return {"hypotheses": hyps, "python_code": str(code)}
        return data


FEATURE_ARCHITECT_SYSTEM_PROMPT = """Ты — ведущий Feature Engineer в команде (Agent 2: FeatureArchitect).
Твоя цель — проанализировать словарь признаков (data_dictionary.csv) и текущую постановку задачи, после чего предложить от 3 до 6 мощных доменных признаков.

ПРАВИЛА СОЗДАНИЯ ПРИЗНАКОВ (по PIPELINE.md §2.4):
1. Используй только детерминированные воспроизводимые преобразования:
   - Отношения величин (ОБЯЗАТЕЛЬНО с защитой от деления на ноль: `+ 1` или `+ 1e-5`).
   - Разности взаимосвязанных величин (например, max - min, start - end).
   - Скорости и интенсивности (величина / (duration + 0.001)).
   - Логарифмирование неотрицательных величин `np.log1p(df[col].clip(lower=0))`.
   - Взаимодействия (произведения связанных факторов).
2. СТРОГИЙ ЗАПРЕТ НА УТЕЧКИ (DATA LEAKAGE):
   - Запрещено использовать или упоминать целевую колонку (target)!
   - Запрещено использовать внешние данные или статистику из test в train.
3. КОД ДОЛЖЕН БЫТЬ ОФОРМЛЕН КАК ФУНКЦИЯ:
   def add_features(df: pd.DataFrame) -> pd.DataFrame:
       df = df.copy()
       # создание новых колонок
       return df

Формат вывода JSON:
{
  "hypotheses": [
    {"feature_name": "ratio_1", "rationale": "смысл", "formula_expression": "col1 / col2"}
  ],
  "python_code": "def add_features(df:\\n    df = df.copy()\\n    ...\\n    return df"
}
"""


# =============================================================================
# 3. СХЕМЫ АГЕНТА 4: CriticMonitor (Монитор качества по PIPELINE.md §8)
# =============================================================================

class CriticVerdict(BaseModel):
    """Вердикт качества эксперимента на основе 5-Fold кросс-валидации."""
    status: Literal["PASS", "FAIL", "UNCERTAIN"] = Field(
        description="PASS — метрика улучшилась, FAIL — деградация или переобучение, UNCERTAIN — шум"
    )
    current_cv_score: float = Field(description="Валидационный скор текущей итерации")
    previous_best_score: float = Field(description="Лучший исторический валидационный скор")
    delta: float = Field(description="Изменение метрики относительно предыдущего лучшего")
    is_improvement: bool = Field(description="Флаг: является ли текущий результат новым рекордом")
    critique_for_agents: str = Field(
        description="Замечания и рекомендации для следующей итерации (почему фичи сработали или нет)"
    )
