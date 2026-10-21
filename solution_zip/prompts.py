from typing import Literal, Optional
from pydantic import BaseModel, Field


class TaskProfile(BaseModel):
    """Структурированный профиль задачи машинного обучения, извлеченный агентом."""
    task_type: Literal["regression", "binary_classification", "multiclass_classification"] = Field(
        description="Тип ML-задачи: регрессия, бинарная классификация или многоклассовая классификация"
    )
    target_column: str = Field(
        description="Название целевой колонки (target) в обучающей выборке"
    )
    id_column: str = Field(
        default="id",
        description="Название колонки с идентификатором строки"
    )
    metric: str = Field(
        description="Основная целевая метрика оптимизации (например, RMSE, Macro F1, ROC-AUC, Accuracy)"
    )
    objective: Literal["minimize", "maximize"] = Field(
        default="minimize",
        description="Направление оптимизации: minimize (для RMSE/MSE/loss) или maximize (для F1/ROC-AUC/Accuracy)"
    )
    description_summary: Optional[str] = Field(
        default="",
        description="Краткая формулировка физического/бизнес-смысла задачи"
    )


PROFILER_SYSTEM_PROMPT = """Ты — ведущий эксперт по машинному обучению и Data Science.
Твоя задача — внимательно изучить условие ML-задачи (task.md) и структуру колонок датасета, после чего точно определить:
1. task_type: "regression", "binary_classification" или "multiclass_classification".
2. target_column: точное имя колонки, которую нужно предсказать.
3. id_column: колонка с идентификатором строки (обычно "id").
4. metric: целевая метрика (например, "RMSE" или "Macro F1").
5. objective: "minimize" (для ошибок вроде RMSE/MSE) или "maximize" (для F1, точности, ROC-AUC).

Ответь СТРОГО валидным JSON-объектом в формате схемы TaskProfile без лишних рассуждений и текста.
"""
