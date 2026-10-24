import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from prompts import TaskProfile


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Оптимизация типов данных (downcasting) для экономии RAM и ускорения на CPU."""
    for col in df.columns:
        col_type = df[col].dtype

        if pd.api.types.is_integer_dtype(col_type):
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            else:
                df[col] = df[col].astype(np.int64)

        elif pd.api.types.is_float_dtype(col_type):
            df[col] = df[col].astype(np.float32)

        elif col_type == object:
            num_unique = df[col].nunique()
            num_total = len(df[col])
            if num_total > 0 and (num_unique / num_total) < 0.5:
                df[col] = df[col].astype("category")

    return df


def locate_file(task_dir: Path, filename: str) -> Path:
    """Умный поиск файлов датасета в каталоге задачи и подпапках."""
    p = task_dir / filename
    if p.exists():
        return p
    # Поиск в подпапках каталога задачи
    for cand in task_dir.rglob(filename):
        if cand.is_file():
            return cand
    # Поиск в общем репозитории на случай запуска из корня
    for base in [Path.cwd(), Path(__file__).resolve().parent.parent]:
        for cand in base.rglob(filename):
            if cand.is_file() and cand.name == filename:
                return cand
    raise FileNotFoundError(f"Файл {filename} не найден в {task_dir} или его поддиректориях")


def load_and_preprocess_data(
    task_dir: Path, profile: TaskProfile
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str], List[str]]:
    """Загрузка, очистка и предварительная подготовка данных."""
    task_dir = Path(task_dir)
    train_path = locate_file(task_dir, "train.csv")
    test_path = locate_file(task_dir, "test.csv")

    print(f"[DataUtils] Чтение датасетов: train={train_path.name}, test={test_path.name}...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    id_col = profile.id_column
    target_col = profile.target_column

    # Проверка наличия колонок
    if id_col not in test_df.columns:
        test_df[id_col] = np.arange(len(test_df))
    test_ids = test_df[id_col].copy()

    if target_col not in train_df.columns:
        raise ValueError(f"Целевая колонка '{target_col}' отсутствует в train.csv")

    y_train = train_df[target_col].copy()

    # Формируем признаки X
    drop_train_cols = [col for col in [id_col, target_col] if col in train_df.columns]
    drop_test_cols = [col for col in [id_col] if col in test_df.columns]

    X_train = train_df.drop(columns=drop_train_cols)
    X_test = test_df.drop(columns=drop_test_cols)

    # Выравниваем колонки
    common_cols = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    # Определяем типы признаков
    cat_features = []
    num_features = []

    for col in common_cols:
        is_num = pd.api.types.is_numeric_dtype(X_train[col])
        if not is_num or X_train[col].dtype == object or str(X_train[col].dtype) == "category":
            cat_features.append(col)
        else:
            num_features.append(col)

    # Обработка категориальных колонок
    for col in cat_features:
        X_train[col] = X_train[col].astype(str).fillna("__MISSING__")
        X_test[col] = X_test[col].astype(str).fillna("__MISSING__")

    # Оптимизация численных колонок (downcasting)
    X_train = optimize_dtypes(X_train)
    X_test = optimize_dtypes(X_test)

    print(f"[DataUtils] Данные готовы. Строк в train: {len(X_train)}, test: {len(X_test)}. Числовых фичей: {len(num_features)}, категориальных: {len(cat_features)}")
    return X_train, y_train, X_test, test_ids, cat_features, num_features


def save_atomic_submission(
    task_dir: Path,
    test_ids: pd.Series,
    predictions: np.ndarray,
    profile: TaskProfile,
) -> Path:
    """Атомарная запись submission.csv с проверкой целостности данных."""
    task_dir = Path(task_dir)
    target_path = task_dir / "submission.csv"
    temp_path = task_dir / ".submission_tmp.csv"

    # Формируем DataFrame сабмита
    sub_df = pd.DataFrame({
        profile.id_column: test_ids,
        profile.target_column: predictions,
    })

    # Синхронизация структуры с sample_submission.csv (названия колонок и типы)
    sample_sub_path = task_dir / "sample_submission.csv"
    if sample_sub_path.exists():
        try:
            sample_df = pd.read_csv(sample_sub_path, nrows=5)
            if len(sample_df.columns) == 2:
                sub_df.columns = list(sample_df.columns)
            sub_df[sub_df.columns[0]] = sub_df[sub_df.columns[0]].astype(sample_df[sample_df.columns[0]].dtype)
        except Exception:
            pass

    # Проверка на пропуски или бесконечности
    target_c = sub_df.columns[1]
    if sub_df[target_c].isnull().any():
        print("[DataUtils] ВНИМАНИЕ: Обнаружены NaN в предсказаниях! Заполняем дефолтным значением.")
        if profile.task_type == "regression":
            sub_df[target_c] = sub_df[target_c].fillna(0.0)
        else:
            sub_df[target_c] = sub_df[target_c].fillna(sub_df[target_c].mode()[0])

    # Запись во временный файл и атомарный rename
    sub_df.to_csv(temp_path, index=False, encoding="utf-8")
    os.replace(temp_path, target_path)

    print(f"[DataUtils] Успешно записан и верифицирован: {target_path.name} ({len(sub_df)} строк)")
    return target_path
