import json
import os
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from prompts import PROFILER_SYSTEM_PROMPT, TaskProfile
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential


def load_env_fallback():
    """Чтение .env без зависимости от python-dotenv."""
    env_paths = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    ]
    for env_path in env_paths:
        if env_path.exists():
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip("'\"")
                            if k and k not in os.environ:
                                os.environ[k] = v
                break
            except Exception:
                pass


load_env_fallback()


def heuristic_task_profile(task_dir: Path) -> TaskProfile:
    """Детерминированное определение профиля задачи без LLM на случай сбоя API."""
    task_dir = Path(task_dir)
    task_md_path = task_dir / "task.md"
    sample_sub_path = task_dir / "sample_submission.csv"
    train_path = task_dir / "train.csv"

    task_text = ""
    if task_md_path.exists():
        try:
            task_text = task_md_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            pass

    # 1. Определяем id_col и target_col по sample_submission.csv
    id_col = "id"
    target_col = None
    if sample_sub_path.exists():
        try:
            sample_df = pd.read_csv(sample_sub_path, nrows=2)
            cols = list(sample_df.columns)
            if len(cols) >= 2:
                id_col = cols[0]
                target_col = cols[1]
        except Exception:
            pass

    # 2. Если target_col не найден, ищем разницу колонок train.csv и test.csv
    test_path = task_dir / "test.csv"
    if not target_col and train_path.exists() and test_path.exists():
        try:
            train_cols = set(pd.read_csv(train_path, nrows=0).columns)
            test_cols = set(pd.read_csv(test_path, nrows=0).columns)
            diff = list(train_cols - test_cols)
            if diff:
                target_col = diff[0]
        except Exception:
            pass

    if not target_col:
        target_col = "target"

    # 3. Определяем task_type по данным train.csv
    task_type = "regression"
    objective = "minimize"
    metric = "RMSE"

    if train_path.exists():
        try:
            sample_train = pd.read_csv(train_path, usecols=[target_col], nrows=2000)
            target_series = sample_train[target_col].dropna()
            n_unique = target_series.nunique()
            is_numeric = pd.api.types.is_numeric_dtype(target_series)

            if not is_numeric or target_series.dtype == object or str(target_series.dtype) == "category":
                if n_unique == 2:
                    task_type = "binary_classification"
                    metric = "ROC-AUC"
                    objective = "maximize"
                else:
                    task_type = "multiclass_classification"
                    metric = "Macro F1"
                    objective = "maximize"
            elif n_unique == 2 and set(target_series.unique()).issubset({0, 1}):
                task_type = "binary_classification"
                metric = "ROC-AUC"
                objective = "maximize"
            elif n_unique <= 20 and is_numeric and (target_series % 1 == 0).all():
                task_type = "multiclass_classification"
                metric = "Macro F1"
                objective = "maximize"
            else:
                task_type = "regression"
                metric = "RMSE"
                objective = "minimize"
        except Exception:
            pass

    # 4. Уточняем метрику по тексту task.md
    task_text_lower = task_text.lower()
    if "macro f1" in task_text_lower or "f1_score" in task_text_lower:
        metric = "Macro F1"
        objective = "maximize"
    elif "rmse" in task_text_lower:
        metric = "RMSE"
        objective = "minimize"
    elif "mae" in task_text_lower:
        metric = "MAE"
        objective = "minimize"
    elif "roc-auc" in task_text_lower or "roc_auc" in task_text_lower:
        metric = "ROC-AUC"
        objective = "maximize"

    return TaskProfile(
        task_type=task_type,
        target_column=target_col,
        id_column=id_col,
        metric=metric,
        objective=objective,
        description_summary="Heuristic fallback profile",
    )


def build_llm_client():
    """Создает клиент OpenAI с учетом переменных окружения платформы."""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    base_url = os.environ.get("OPENAI_BASE_URL", None)
    if base_url:
        base_url = base_url.strip()

    if not api_key:
        return None

    try:
        from openai import OpenAI
        return OpenAI(api_key=api_key, base_url=base_url)
    except Exception as e:
        print(f"[TaskProfiler] Не удалось инициализировать OpenAI клиент: {e}")
        return None


def get_task_profile(task_dir: Path) -> TaskProfile:
    """Извлекает профиль задачи через LLM с автоматическим fallback на эвристику."""
    task_dir = Path(task_dir)
    client = build_llm_client()
    model = os.environ.get("OPENAI_MODEL", "qwen")

    if not client:
        print("[TaskProfiler] OpenAI API key не задан или библиотека openai не установлена, используется детерминированная эвристика.")
        return heuristic_task_profile(task_dir)

    task_md_path = task_dir / "task.md"
    task_text = ""
    if task_md_path.exists():
        task_text = task_md_path.read_text(encoding="utf-8", errors="ignore")[:4000]

    dict_path = task_dir / "data_dictionary.csv"
    dict_text = ""
    if dict_path.exists():
        try:
            df_dict = pd.read_csv(dict_path, nrows=20)
            dict_text = df_dict.to_string()
        except Exception:
            pass

    prompt = f"""Ниже приведено описание ML-задачи и словарь признаков.
=== TASK.MD ===
{task_text}

=== DATA DICTIONARY (FIRST 20 ROWS) ===
{dict_text}
"""

    import openai

    @retry(
        wait=wait_random_exponential(multiplier=1, max=30),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError)),
        reraise=True,
    )
    def call_model():
        print(f"[TaskProfiler] Запрос к LLM ({model})...")
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": PROFILER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=2500,
        )

    try:
        response = call_model()
        content = response.choices[0].message.content
        data = json.loads(content)
        profile = TaskProfile(**data)
        print(f"[TaskProfiler] Профиль успешно определен LLM: {profile.task_type}, target='{profile.target_column}', metric='{profile.metric}'")
        return profile
    except Exception as e:
        print(f"[TaskProfiler] Ошибка при обращении к LLM ({e}), переключение на детерминированную эвристику.")
        return heuristic_task_profile(task_dir)
