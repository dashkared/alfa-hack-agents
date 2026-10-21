import os
import sys
import time
from pathlib import Path


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


# Загружаем переменные окружения
load_env_fallback()

from data_utils import load_and_preprocess_data, save_atomic_submission
from ml_models import generate_failsafe_baseline, train_gradient_boosting_cv
from task_profiler import get_task_profile


def resolve_task_path(arg_path: str = None) -> Path:
    """Умное определение пути к задаче для локальных тестов и продакшена."""
    if arg_path:
        p = Path(arg_path).resolve()
        if p.is_dir():
            p = p / "task.md"
        if p.exists():
            return p

    # 1. Проверяем task.md в текущей директории (стандарт платформы Boosters)
    cwd_task = Path.cwd() / "task.md"
    if cwd_task.exists():
        return cwd_task

    # 2. Локальный поиск по известным папкам датасетов
    candidates = [
        Path("data_for_participants/Регрессия/task.md"),
        Path("data_for_participants/Многоклассовая классификация/task.md"),
        Path(__file__).resolve().parent.parent / "data_for_participants/Регрессия/task.md",
        Path(__file__).resolve().parent.parent / "data_for_participants/Многоклассовая классификация/task.md",
    ]
    for cand in candidates:
        if cand.exists():
            print(f"[Launcher] Путь к задаче не был указан. Автоматически выбран: {cand}")
            return cand.resolve()

    return Path("task.md").resolve()


def main(task_md_path: Path):
    start_time = time.time()
    task_md_path = resolve_task_path(str(task_md_path) if task_md_path else None)
    task_dir = task_md_path.parent

    print("=" * 70)
    print("🚀 ALFA-HACK: AI-AGENT AUTOML PIPELINE (MVP)")
    print(f"📁 Директория задачи: {task_dir}")
    print(f"📄 Файл задачи: {task_md_path.name}")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # ШАГ 1: Извлечение профиля задачи (LLM с fallback на эвристику)
    # -------------------------------------------------------------------------
    print("\n[1/5] Профайлинг задачи...")
    profile = get_task_profile(task_dir)
    print(f"      Тип задачи: {profile.task_type}")
    print(f"      Целевая колонка: {profile.target_column}")
    print(f"      ID колонка: {profile.id_column}")
    print(f"      Метрика: {profile.metric} ({profile.objective})")

    # -------------------------------------------------------------------------
    # ШАГ 2: Загрузка данных, валидация и сжатие типов
    # -------------------------------------------------------------------------
    print("\n[2/5] Загрузка и оптимизация данных...")
    X_train, y_train, X_test, test_ids, cat_features, num_features = load_and_preprocess_data(
        task_dir, profile
    )

    # -------------------------------------------------------------------------
    # ШАГ 3: Защитный Fail-Safe сабмит (гарантия ненулевого результата)
    # -------------------------------------------------------------------------
    print("\n[3/5] Создание первичного Fail-Safe сабмита...")
    failsafe_preds = generate_failsafe_baseline(y_train, test_ids, profile, task_dir)

    # -------------------------------------------------------------------------
    # ШАГ 4: Обучение модели градиентного бустинга с 5-Fold кросс-валидацией
    # -------------------------------------------------------------------------
    print("\n[4/5] Обучение ансамбля моделей на 8 CPU...")
    try:
        oof_score, model_preds, backend = train_gradient_boosting_cv(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            cat_features=cat_features,
            num_features=num_features,
            profile=profile,
            n_splits=5,
        )

        # ---------------------------------------------------------------------
        # ШАГ 5: Атомарное обновление submission.csv лучшими предсказаниями
        # ---------------------------------------------------------------------
        print("\n[5/5] Обновление submission.csv предсказаниями обученной модели...")
        save_atomic_submission(task_dir, test_ids, model_preds, profile)
        print(f"      ✅ Сабмит успешно обновлен моделью '{backend}'!")
        print(f"      🎯 Валидационный CV-скор ({profile.metric}): {oof_score:.4f}")

    except Exception as e:
        print(f"\n⚠️ Ошибка в процессе обучения модели: {e}")
        print("      Защитный Fail-Safe сабмит сохранен и защищает команду от нулевого балла!")

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"🏁 Пайплайн успешно завершил работу за {elapsed:.1f} сек. ({elapsed/60:.2f} мин.)")
    print(f"📦 submission.csv готов: {task_dir / 'submission.csv'}")
    print("=" * 70)


if __name__ == "__main__":
    task_arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(task_arg)
