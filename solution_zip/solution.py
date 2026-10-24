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
MAX_LLM_FEATURE_ITERATIONS = 2
TIME_BUDGET_SECONDS = 9600  # 2.6 часа (лимит платформы 3 часа)
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
    print("🤖 ALFA-HACK: MULTI-AGENT AUTORESEARCH & ENSEMBLING SYSTEM (MAS 2.0)")
    print(f"📁 Рабочий каталог: {task_dir}")
    print(f"📄 Паспорт задачи:  {task_md_path.name}")
    print("⚙️ Пайплайн: Universal Features + LLM Hypotheses + Pruning + Big-3 Portfolio (CB/LGB/XGB) + Blending")
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

    # Адаптивный базовый оценщик: для многоклассовой классификации используем быстрый
    # и точный LightGBM (предотвращает ложные откаты и ускоряет цикл в 4 раза),
    # для регрессии — CatBoost.
    base_model = "lightgbm" if profile.task_type == "multiclass_classification" else "catboost"
    base_model_label = "LightGBM" if base_model == "lightgbm" else "CatBoost"

    print("\n" + "=" * 80)
    print("🔄 ЗАПУСК ЦИКЛА AUTORESEARCH: FEATURE & MODEL SPACE EXPLORATION")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Итерация 0: Оценка чистого бейзлайна
    # -------------------------------------------------------------------------
    print(f"\n[Итерация 0: Baseline Model ({base_model_label})]")
    score_0, preds_0, backend_0 = model_agent.train_and_evaluate(
        X_train_raw, y_train, X_test_raw, cat_features, model_family=base_model
    )
    verdict_0 = critic.evaluate_experiment(
        iteration=0,
        hypothesis_name=f"Baseline Model ({base_model_label})",
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
    # ФАЗА 1: Фундаментальные табличные признаки (Universal Tabular Engine)
    # -------------------------------------------------------------------------
    print(f"\n[Итерация {current_iter}: Universal Tabular Feature Engineering (Агрегации, Частоты, Пропуски)]")
    try:
        X_tr_univ, X_te_univ, cats_univ = feature_agent.apply_universal_tabular_features(
            best_X_train, best_X_test, best_cat_cols
        )
        score_univ, preds_univ, _ = model_agent.train_and_evaluate(
            X_tr_univ, y_train, X_te_univ, cats_univ, model_family=base_model
        )
        verdict_univ = critic.evaluate_experiment(
            iteration=current_iter,
            hypothesis_name="Universal Tabular Engine (Group Stats & Freq)",
            current_cv_score=score_univ,
            test_predictions=preds_univ,
            test_ids=test_ids,
        )
        print(f"  -> Вердикт Монитора: [{verdict_univ.status}] | CV {profile.metric}: {verdict_univ.current_cv_score:.4f} (Δ = {verdict_univ.delta:+.4f})")
        print(f"  -> {verdict_univ.critique_for_agents}")

        if verdict_univ.status == "PASS":
            best_X_train = X_tr_univ
            best_X_test = X_te_univ
            best_cat_cols = cats_univ
            last_feedback = verdict_univ.critique_for_agents
        else:
            print(f"  🔄 [Rollback] Откат к состоянию бейзлайна.")
            last_feedback = verdict_univ.critique_for_agents
    except Exception as e:
        print(f"  ⚠️ Ошибка в Universal Tabular Features: {e}")

    current_iter += 1

    # -------------------------------------------------------------------------
    # ФАЗА 2: LLM FeatureArchitect (Доменные гипотезы и формулы)
    # -------------------------------------------------------------------------
    for feat_iter in range(1, MAX_LLM_FEATURE_ITERATIONS + 1):
        elapsed_sec = time.time() - start_time
        if elapsed_sec > TIME_BUDGET_SECONDS:
            print(f"\n⏰ [Coordinator] Превышен бюджет времени. Переход к ансамблированию.")
            break

        stop_needed, stop_reason = critic.should_stop()
        if stop_needed:
            print(f"\nℹ️ [Coordinator] Признаковое пространство стабилизировалось ({stop_reason}).")
            break

        print(f"\n[Итерация {current_iter}: FeatureArchitect Agent (LLM Гипотеза #{feat_iter})]")
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

            print(f"  -> Обучение {base_model_label} на кандидате признаков (Всего колонок: {len(common_cols)})...")
            score_i, preds_i, backend_i = model_agent.train_and_evaluate(
                X_train_cand, y_train, X_test_cand, cand_cats, model_family=base_model
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
    # ФАЗА 3: Очистка признаков (Feature Pruning)
    # -------------------------------------------------------------------------
    print(f"\n[Итерация {current_iter}: ModelPortfolio Agent (Очистка неинформативных фичей)]")
    try:
        pruned_X_tr, pruned_X_te, pruned_cats, dropped = model_agent.prune_features(
            best_X_train, best_X_test, y_train, best_cat_cols
        )
        if dropped:
            score_pruned, preds_pruned, _ = model_agent.train_and_evaluate(
                pruned_X_tr, y_train, pruned_X_te, pruned_cats, model_family=base_model
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

    # Сбрасываем пул моделей перед обучением разнородных архитектур на лучшем наборе признаков
    model_agent.clear_pool()
    # Регистрируем базовую модель на лучшем пространстве признаков
    print(f"\n[ModelPortfolio] Фиксация базовой модели {base_model_label} на лучшем пространстве признаков...")
    model_agent.train_and_evaluate(
        best_X_train, y_train, best_X_test, best_cat_cols, model_family=base_model
    )

    # -------------------------------------------------------------------------
    # ФАЗА 4: Исследование пространства моделей («Большая тройка»: CatBoost, LightGBM, XGBoost)
    # -------------------------------------------------------------------------
    other_models = ["xgboost", "catboost"] if base_model == "lightgbm" else ["lightgbm", "xgboost"]

    for m_fam in other_models:
        fam_label = "LightGBM" if m_fam == "lightgbm" else ("XGBoost" if m_fam == "xgboost" else "CatBoost")
        print(f"\n[Итерация {current_iter}: ModelPortfolio Agent (Архитектура: {fam_label})]")
        try:
            score_m, preds_m, backend_m = model_agent.train_and_evaluate(
                best_X_train, y_train, best_X_test, best_cat_cols, model_family=m_fam
            )
            verdict_m = critic.evaluate_experiment(
                iteration=current_iter,
                hypothesis_name=f"Model Diversity: {fam_label}",
                current_cv_score=score_m,
                test_predictions=preds_m,
                test_ids=test_ids,
            )
            print(f"  -> Вердикт Монитора: [{verdict_m.status}] | CV {profile.metric}: {verdict_m.current_cv_score:.4f} (Δ = {verdict_m.delta:+.4f})")
        except Exception as e:
            print(f"  ⚠️ Ошибка обучения {fam_label}: {e}")

        current_iter += 1

    # -------------------------------------------------------------------------
    # ФАЗА 5: Ансамблирование (Ensembling / Blending Agent)
    # Поиск оптимальных весов смешивания CatBoost + LightGBM + XGBoost
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
