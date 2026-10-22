import os
import re
import sys
import time
from pathlib import Path

# Добавляем текущую директорию скрипта в sys.path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))


def load_env_fallback():
    """Чтение .env без зависимости от сторонних библиотек."""
    possible_envs = [
        Path.cwd() / ".env",
        CURRENT_DIR / ".env",
        CURRENT_DIR.parent / ".env",
    ]
    for env_path in possible_envs:
        if env_path.exists():
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip().strip("'\"")
                        if k not in os.environ:
                            os.environ[k] = v
            break


load_env_fallback()

from critic_agent import CriticMonitorAgent
from data_utils import (
    load_and_preprocess_data,
    optimize_dtypes,
)
from feature_agent import FeatureArchitectAgent
from ml_models import generate_failsafe_baseline
from model_agent import ModelPortfolioAgent
from task_profiler import get_task_profile

# Настройки цикла AutoResearch
MAX_FEATURE_ITERATIONS = 2
TIME_BUDGET_SECONDS = 9000  # 2.5 часа (лимит платформы 3 часа)
PATIENCE = 2


def resolve_task_path(arg_path: str = None) -> Path:
    """Универсальное разрешение пути к task.md."""
    if arg_path:
        p = Path(arg_path).resolve()
        if p.is_dir():
            p = p / "task.md"
        if p.exists():
            return p

    cwd_task = Path.cwd() / "task.md"
    if cwd_task.exists():
        return cwd_task

    candidates = [
        Path("data_for_participants/Регрессия/task.md"),
        Path("data_for_participants/Многоклассовая классификация/task.md"),
        Path(__file__).resolve().parent.parent / "data_for_participants/Регрессия/task.md",
        Path(__file__).resolve().parent.parent / "data_for_participants/Многоклассовая классификация/task.md",
    ]
    for cand in candidates:
        if cand.exists():
            print(f"[Launcher] Автоматически выбран демо-датасет: {cand}")
            return cand.resolve()

    return Path("task.md").resolve()


def main(task_md_path: Path):
    start_time = time.time()
    task_md_path = resolve_task_path(str(task_md_path) if task_md_path else None)
    task_dir = task_md_path.parent

    print("=" * 80)
    print("🤖 ALFA-HACK: MULTI-AGENT AUTORESEARCH & ENSEMBLING SYSTEM (MAS)")
    print(f"📁 Рабочий каталог: {task_dir}")
    print(f"📄 Паспорт задачи:  {task_md_path.name}")
    print("⚙️ Пайплайн: Feature Engineering + Pruning + Portfolio (CatBoost/HistGB/ExtraTrees) + Blending")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # ШАГ 1: Агент 1 (TaskProfiler) — Формирование паспорта эксперимента
    # -------------------------------------------------------------------------
    print("\n[Агент 1: TaskProfiler] Анализ требований и словаря данных...")
    profile = get_task_profile(task_dir)
    print(f"      • Тип задачи:       {profile.task_type}")
    print(f"      • Целевой признак:  {profile.target_column}")
    print(f"      • ID колонка:       {profile.id_column}")
    print(f"      • Целевая метрика:  {profile.metric} ({profile.objective})")

    # -------------------------------------------------------------------------
    # ШАГ 2: Загрузка и оптимизация данных (Data Compressor)
    # -------------------------------------------------------------------------
    print("\n[Data Compressor] Загрузка и сжатие типов данных (Downcasting)...")
    X_train_raw, y_train, X_test_raw, test_ids, cat_features, num_features = load_and_preprocess_data(
        task_dir, profile
    )

    # -------------------------------------------------------------------------
    # ШАГ 3: Правило первой минуты — Защитный Fail-Safe сабмит
    # -------------------------------------------------------------------------
    print("\n[Gatekeeper] Создание защитного сабмита 1-й минуты...")
    generate_failsafe_baseline(y_train, test_ids, profile, task_dir)

    # -------------------------------------------------------------------------
    # ШАГ 4: Инициализация агентов
    # -------------------------------------------------------------------------
    critic = CriticMonitorAgent(task_dir, profile, patience=PATIENCE)
    feature_agent = FeatureArchitectAgent(task_dir, profile)
    model_agent = ModelPortfolioAgent(profile, n_splits=5)

    print("\n" + "=" * 80)
    print("🔄 ЗАПУСК ЦИКЛА AUTORESEARCH: FEATURE & MODEL SPACE EXPLORATION")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Итерация 0: Оценка чистого бейзлайна (CatBoost)
    # -------------------------------------------------------------------------
    print("\n[Итерация 0: Baseline Model (CatBoost)]")
    score_0, preds_0, backend_0 = model_agent.train_and_evaluate(
        X_train_raw, y_train, X_test_raw, cat_features, model_family="catboost"
    )
    verdict_0 = critic.evaluate_experiment(
        iteration=0,
        hypothesis_name="Baseline Model (CatBoost)",
        current_cv_score=score_0,
        test_predictions=preds_0,
        test_ids=test_ids,
    )
    print(f"  -> Вердикт Монитора: [{verdict_0.status}] | CV {profile.metric}: {verdict_0.current_cv_score:.4f}")

    best_X_train = X_train_raw.copy()
    best_X_test = X_test_raw.copy()
    best_cat_cols = cat_features.copy()
    last_feedback = verdict_0.critique_for_agents

    current_iter = 1

    # -------------------------------------------------------------------------
    # ФАЗА 1: Feature Engineering (Генерация и отбор доменных признаков)
    # -------------------------------------------------------------------------
    for feat_iter in range(1, MAX_FEATURE_ITERATIONS + 1):
        elapsed_sec = time.time() - start_time
        if elapsed_sec > TIME_BUDGET_SECONDS:
            print(f"\n⏰ [Coordinator] Превышен бюджет времени. Переход к ансамблированию.")
            break

        stop_needed, stop_reason = critic.should_stop()
        if stop_needed:
            print(f"\nℹ️ [Coordinator] Признаковое пространство стабилизировалось ({stop_reason}).")
            break

        print(f"\n[Итерация {current_iter}: FeatureArchitect Agent (Гипотеза #{feat_iter})]")
        transform_fn, hyp_name = feature_agent.generate_feature_transform(
            best_X_train, critique_feedback=last_feedback
        )

        try:
            X_train_cand = transform_fn(best_X_train)
            X_test_cand = transform_fn(best_X_test)

            common_cols = [c for c in X_train_cand.columns if c in X_test_cand.columns]
            X_train_cand = optimize_dtypes(X_train_cand[common_cols])
            X_test_cand = optimize_dtypes(X_test_cand[common_cols])
            cand_cats = [c for c in best_cat_cols if c in common_cols]

            print(f"  -> Обучение CatBoost на кандидате признаков (Всего колонок: {len(common_cols)})...")
            score_i, preds_i, backend_i = model_agent.train_and_evaluate(
                X_train_cand, y_train, X_test_cand, cand_cats, model_family="catboost"
            )

            verdict_i = critic.evaluate_experiment(
                iteration=current_iter,
                hypothesis_name=hyp_name,
                current_cv_score=score_i,
                test_predictions=preds_i,
                test_ids=test_ids,
            )

            print(f"  -> Вердикт Монитора: [{verdict_i.status}] | CV {profile.metric}: {verdict_i.current_cv_score:.4f} (Δ = {verdict_i.delta:+.4f})")
            print(f"  -> {verdict_i.critique_for_agents}")

            if verdict_i.status == "PASS":
                best_X_train = X_train_cand
                best_X_test = X_test_cand
                best_cat_cols = cand_cats
                last_feedback = verdict_i.critique_for_agents
            else:
                print(f"  🔄 [Rollback] Откат к состоянию итерации {critic.best_iteration} (CV {profile.metric}: {critic.best_score:.4f})")
                last_feedback = verdict_i.critique_for_agents

        except Exception as e:
            print(f"  ⚠️ Ошибка на итерации {current_iter}: {e}")
            last_feedback = f"Ошибка генерации: {e}. Попробуй другие признаки."

        current_iter += 1

    # -------------------------------------------------------------------------
    # ФАЗА 2: Очистка признаков (Feature Pruning)
    # -------------------------------------------------------------------------
    print(f"\n[Итерация {current_iter}: ModelPortfolio Agent (Очистка неинформативных фичей)]")
    try:
        pruned_X_tr, pruned_X_te, pruned_cats, dropped = model_agent.prune_features(
            best_X_train, best_X_test, y_train, best_cat_cols
        )
        if dropped:
            score_pruned, preds_pruned, _ = model_agent.train_and_evaluate(
                pruned_X_tr, y_train, pruned_X_te, pruned_cats, model_family="catboost"
            )
            verdict_pruned = critic.evaluate_experiment(
                iteration=current_iter,
                hypothesis_name=f"Feature Pruning (Удалено {len(dropped)} шумных фичей)",
                current_cv_score=score_pruned,
                test_predictions=preds_pruned,
                test_ids=test_ids,
            )
            print(f"  -> Вердикт Монитора: [{verdict_pruned.status}] | CV {profile.metric}: {verdict_pruned.current_cv_score:.4f} (Δ = {verdict_pruned.delta:+.4f})")
            if verdict_pruned.status == "PASS":
                best_X_train, best_X_test, best_cat_cols = pruned_X_tr, pruned_X_te, pruned_cats
            else:
                print(f"  🔄 [Rollback] Очистка не улучшила скор, сохранен полный набор фичей.")
    except Exception as e:
        print(f"  ⚠️ Ошибка при очистке признаков: {e}")

    current_iter += 1

    # -------------------------------------------------------------------------
    # ФАЗА 3: Исследование пространства моделей (Model Portfolio)
    # Обучение HistGradientBoosting и ExtraTrees на лучших фичах
    # -------------------------------------------------------------------------
    print(f"\n[Итерация {current_iter}: ModelPortfolio Agent (Архитектура: HistGradientBoosting)]")
    try:
        score_hist, preds_hist, backend_hist = model_agent.train_and_evaluate(
            best_X_train, y_train, best_X_test, best_cat_cols, model_family="hist_gb"
        )
        verdict_hist = critic.evaluate_experiment(
            iteration=current_iter,
            hypothesis_name="Model Diversity: HistGradientBoosting",
            current_cv_score=score_hist,
            test_predictions=preds_hist,
            test_ids=test_ids,
        )
        print(f"  -> Вердикт Монитора: [{verdict_hist.status}] | CV {profile.metric}: {verdict_hist.current_cv_score:.4f} (Δ = {verdict_hist.delta:+.4f})")
    except Exception as e:
        print(f"  ⚠️ Ошибка обучения HistGradientBoosting: {e}")

    current_iter += 1

    print(f"\n[Итерация {current_iter}: ModelPortfolio Agent (Архитектура: ExtraTrees Bagging)]")
    try:
        score_et, preds_et, backend_et = model_agent.train_and_evaluate(
            best_X_train, y_train, best_X_test, best_cat_cols, model_family="extra_trees"
        )
        verdict_et = critic.evaluate_experiment(
            iteration=current_iter,
            hypothesis_name="Model Diversity: ExtraTrees (Bagging)",
            current_cv_score=score_et,
            test_predictions=preds_et,
            test_ids=test_ids,
        )
        print(f"  -> Вердикт Монитора: [{verdict_et.status}] | CV {profile.metric}: {verdict_et.current_cv_score:.4f} (Δ = {verdict_et.delta:+.4f})")
    except Exception as e:
        print(f"  ⚠️ Ошибка обучения ExtraTrees: {e}")

    current_iter += 1

    # -------------------------------------------------------------------------
    # ФАЗА 4: Ансамблирование (Ensembling / Blending Agent)
    # Поиск оптимальных весов смешивания CatBoost + HistGB + ExtraTrees
    # -------------------------------------------------------------------------
    print(f"\n[Итерация {current_iter}: ModelPortfolio Agent (Multi-Model Blending)]")
    try:
        blend_result = model_agent.blend_portfolio(y_train)
        if blend_result:
            score_blend, preds_blend, blend_name = blend_result
            verdict_blend = critic.evaluate_experiment(
                iteration=current_iter,
                hypothesis_name=blend_name,
                current_cv_score=score_blend,
                test_predictions=preds_blend,
                test_ids=test_ids,
            )
            print(f"  -> Вердикт Монитора: [{verdict_blend.status}] | CV {profile.metric}: {verdict_blend.current_cv_score:.4f} (Δ = {verdict_blend.delta:+.4f})")
            if verdict_blend.status == "PASS":
                print(f"  🏆 Ансамбль превзошел одиночные модели и стал финальным сабмитом!")
            else:
                print(f"  🔄 [Rollback] Ансамбль не дал преимущества, сохранен лучший предикт.")
    except Exception as e:
        print(f"  ⚠️ Ошибка ансамблирования: {e}")

    # -------------------------------------------------------------------------
    # ИТОГОВЫЙ ОТЧЕТ И СВОДНАЯ ТАБЛИЦА ЭКСПЕРИМЕНТОВ
    # -------------------------------------------------------------------------
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("🏆 МУЛЬТИАГЕНТНЫЙ ЦИКЛ ПОЛНОСТЬЮ ЗАВЕРШЕН")
    print(f"⏱️ Общее время работы: {elapsed:.1f} сек. ({elapsed/60:.2f} мин.)")
    print(f"🎯 Итоговый лучший CV-скор ({profile.metric}): {critic.best_score:.4f} (Итерация {critic.best_iteration})")
    print("=" * 80)

    print("\n📊 ЖУРНАЛ ЭКСПЕРИМЕНТОВ (EXPERIMENT LOG):")
    print(f"{'Итер':<6}{'Гипотеза':<45}{'Скор (' + profile.metric + ')':<20}{'Статус':<10}{'Дельта'}")
    print("-" * 92)
    for row in critic.history:
        it, h_name, sc, st, d = row
        print(f"{it:<6}{h_name[:42]:<45}{sc:<20.4f}{st:<10}{d:+.4f}")

    print("\n📦 Проверенный submission.csv сохранен: " + str(task_dir / "submission.csv"))


if __name__ == "__main__":
    task_arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(task_arg)
