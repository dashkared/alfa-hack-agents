"""ALFA HACK — Autonomous AutoML / AutoResearch agent (entry point).

Pipeline (every phase has a deterministic fallback and a time budget):

  1. TaskProfilerAgent   — task type / target / metric / temporal & entity structure;
  2. DataCompressor      — downcasting, NaN policy, fail-safe submission in the first minute;
  3. FeatureArchitect    — feature blocks (base -> freq -> entity history -> crosses -> LLM);
                           every block is accepted only if the Critic confirms a real gain;
  4. ModelPortfolioAgent — expanding-window / stratified CV, heterogeneous portfolio,
                           constrained blending, class-multiplier calibration, full refit;
  5. CriticMonitorAgent  — accept/reject with paired fold comparison (anti metric-hacking).

Never exits non-zero, never leaves the task directory without a valid submission.csv.
"""
import os
import sys
import time
import traceback
from pathlib import Path

# OpenMP runtime / threading hygiene for the Linux evaluation container
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from critic_agent import CriticMonitorAgent  # noqa: E402
from data_utils import (locate_file, make_submission_frame, read_csv_safe,  # noqa: E402
                        save_submission, validate_submission)
from entity_features import MODE_ALL, MODE_BEFORE, MODE_LOO  # noqa: E402
from evaluation import metric_direction  # noqa: E402
from feature_agent import FeatureBundle, LLMFeatureGenerator  # noqa: E402
from model_agent import ModelPortfolioAgent  # noqa: E402
from prompts import TaskProfile  # noqa: E402
from task_profiler import (TaskProfilerAgent, heuristic_task_profile,  # noqa: E402
                           inverse_target, standardize_target)

TIME_BUDGET_SECONDS = int(os.environ.get("SOLUTION_TIME_BUDGET", "9000"))  # 2.5h < 3h limit
RESERVE_SECONDS = 300
SEED = 42
N_JOBS = int(os.environ.get("SOLUTION_N_JOBS", str(max(2, min(8, (os.cpu_count() or 4))))))


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def resolve_task_path(arg_path: str = None) -> Path:
    """Universal resolution of the task.md path (argument / cwd / known demo folders)."""
    if arg_path:
        p = Path(arg_path).expanduser()
        try:
            p = p.resolve()
        except Exception:
            pass
        if p.is_file():
            return p
        if p.is_dir():
            if (p / "task.md").exists():
                return p / "task.md"
            md = sorted(p.glob("*.md"))
            if md:
                return md[0]
            return p / "task.md"
        if p.parent.is_dir():
            return p
    cwd = Path.cwd().resolve()
    if (cwd / "task.md").exists():
        return cwd / "task.md"
    if (cwd / "train.csv").exists():
        return cwd / "task.md"
    for md in sorted(cwd.glob("*.md")):
        if "task" in md.name.lower():
            return md
    # last resort: a sibling directory that actually contains the data (never hard-codes a
    # specific competition — the package must behave identically on any unseen task)
    for sub in sorted(cwd.parent.glob("*/task.md")):
        if (sub.parent / "train.csv").exists():
            print(f"[Launcher] Задача найдена рядом: {sub}")
            return sub.resolve()
    return cwd / "task.md"


def write_blind_failsafe(task_dir: Path, verbose: bool = True) -> bool:
    """Last-resort submission built from sample_submission.csv only (no data needed)."""
    try:
        sample_path = locate_file(task_dir, "sample_submission.csv", required=False)
        if sample_path is None:
            return False
        sample = read_csv_safe(sample_path)
        if sample.shape[1] < 2 or len(sample) == 0:
            return False
        out = sample.copy()
        tgt = out.columns[-1]
        if pd.api.types.is_numeric_dtype(sample[tgt]):
            fill = float(pd.to_numeric(sample[tgt], errors="coerce").median() or 0.0)
            out[tgt] = 0.0 if not np.isfinite(fill) else fill
        else:
            mode = sample[tgt].astype(str).mode()
            out[tgt] = str(mode.iloc[0]) if len(mode) else str(sample[tgt].iloc[0])
        tmp = task_dir / ".blind_submission.csv"
        out.to_csv(tmp, index=False, encoding="utf-8")
        os.replace(tmp, task_dir / "submission.csv")
        if verbose:
            print(f"[FailSafe] Записан защитный сабмит из sample_submission ({len(out)} строк)")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[FailSafe] Не удалось записать защитный сабмит: {exc}")
        return False


def prepare_targets(profile: TaskProfile, train_df: pd.DataFrame):
    """Drop rows with a missing target and return (y_model, train_df, labels)."""
    if profile.task_type == "multilabel":
        cols = profile.target_columns
        keep = train_df[cols].notna().all(axis=1)
        return None, train_df.loc[keep].reset_index(drop=True), cols
    before = len(train_df)
    mask = train_df[profile.target_column].notna()
    if profile.is_classification:
        mask = mask & train_df[profile.target_column].astype(str).ne("nan")
    train_df = train_df.loc[mask].reset_index(drop=True)
    if len(train_df) < before:
        print(f"[DataCompressor] отброшено {before - len(train_df)} строк без целевого значения")
    y, labels = standardize_target(train_df[profile.target_column], profile)
    if labels:
        profile.class_labels = labels
        profile.n_classes = len(labels)
    return y, train_df, labels


def align_test_ids(profile: TaskProfile, train_df: pd.DataFrame, test_df: pd.DataFrame,
                   sample_df) -> pd.Series:
    """Make sure the test frame carries the id column used by sample_submission."""
    id_col = profile.id_column
    if id_col in test_df.columns:
        ids = test_df[id_col]
    elif sample_df is not None and sample_df.shape[1] >= 1:
        ids = sample_df[sample_df.columns[0]]
        test_df[id_col] = ids.values
    else:
        ids = pd.Series(np.arange(len(test_df)))
        test_df[id_col] = ids.values
    return ids.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# main pipeline
# --------------------------------------------------------------------------- #
def main(task_arg: str = None):
    t_start = time.time()
    deadline = t_start + TIME_BUDGET_SECONDS

    def remaining() -> float:
        return deadline - time.time() - RESERVE_SECONDS

    def phase(name: str, budget_s: float) -> float:
        nonlocal deadline
        end = min(deadline, time.time() + max(30.0, budget_s))
        print(f"\n{'=' * 78}\n⏱  {name} (бюджет {max(30, budget_s):.0f}s, всего осталось {remaining():.0f}s)\n{'=' * 78}",
              flush=True)
        return end

    print("=" * 78)
    print("🤖 ALFA-HACK AUTONOMOUS ML AGENT — AutoResearch pipeline")
    print(f"   бюджет времени: {TIME_BUDGET_SECONDS}s, потоков: {N_JOBS}")
    print("=" * 78)

    task_md = resolve_task_path(task_arg)
    task_dir = task_md.parent
    print(f"[Launcher] Каталог задачи: {task_dir}")

    # ---- LAYER 0: blind fail-safe (works even if the CSVs are unreadable) ----
    write_blind_failsafe(task_dir)

    # ---- LAYER 1: profiling ------------------------------------------------
    phase("Фаза 1/6 — профилирование задачи и структуры данных", 240)
    profiler = TaskProfilerAgent(task_dir)
    profile, train_df, test_df, sample_df = profiler.run()
    print("\n[TaskProfiler] Паспорт эксперимента:")
    print(profile.pretty())
    try:
        import json as _json
        print("[TaskProfiler] JSON-роутер: "
              + _json.dumps(profile.router_report(), ensure_ascii=False))
    except Exception:
        pass
    if profile.notes.get("text_mentions_classification") is None or \
            str(profile.metric) in ("RMSE", "Macro F1"):
        # ambiguous passport -> one bounded LLM call is worth it
        try:
            profile = profiler.llm_assist(profile, train_df)
        except Exception as exc:  # noqa: BLE001
            print(f"[TaskProfiler] LLM-ассист пропущен: {exc}")

    # ---- LAYER 2: data preparation + quick fail-safe model ------------------
    y, train_df, labels = prepare_targets(profile, train_df)
    test_ids = align_test_ids(profile, train_df, test_df, sample_df)
    print(f"[DataCompressor] train={train_df.shape}, test={test_df.shape}")

    if profile.task_type == "multilabel":
        return multilabel_pipeline(task_dir, profile, train_df, test_df, test_ids, phase, remaining)

    bundle = FeatureBundle(profile, train_df, test_df)
    agent = ModelPortfolioAgent(profile, n_jobs=N_JOBS, seed=SEED, deadline=remaining)
    critic = CriticMonitorAgent(profile.metric, profile.objective)
    folds = agent.make_folds(y, bundle)
    print(f"[ModelPortfolio] схема валидации: {len(folds)} фолд(ов), "
          f"{'временная (expanding window)' if bundle.periods and len(bundle.periods) >= 3 else 'K-fold'}"
          f"; entity-признаки: {bundle.entity_enabled}")

    # quick model -> immediate valid submission (fail-safe layer 2)
    quick_family = "lgbm" if profile.is_classification else "catboost"
    try:
        quick = agent.run_cv(quick_family, bundle, y, folds, fast=True, entity_mode=MODE_LOO)
        if quick is not None:
            _write_predictions(task_dir, profile, test_ids, quick.get("test_proba"),
                               quick.get("test_pred"), labels, agent, None)
            print("[FailSafe] Быстрая модель записала первый осмысленный submission.csv")
    except Exception as exc:  # noqa: BLE001
        print(f"[FailSafe] быстрая модель не удалась: {exc}")

    # ---- PHASE 3: feature-block AutoResearch loop ---------------------------
    phase("Фаза 3/6 — AutoResearch: отбор блоков признаков", 1200)
    screen_folds = folds[-min(2, len(folds)):] if len(folds) > 2 else folds
    best_state = {"entity": bundle.entity_enabled, "blocks": list(bundle.blocks)}

    # A 400-iteration "cheap" screen can select a feature set the full-capacity models do
    # not reproduce, so when the budget allows we screen with the shipped model capacity.
    screen_fast = remaining() < max(1200.0, 0.35 * TIME_BUDGET_SECONDS)

    def screen(name: str, entity: bool, residual: bool = False, mode: str = MODE_LOO,
               hier: bool = False):
        nonlocal best_state
        prev_entity = bundle.entity_enabled
        prev_hier = bundle.hier_enabled
        bundle.entity_enabled = bool(entity and bundle.entity is not None)
        bundle.hier_enabled = bool(hier and bool(bundle.hier))
        try:
            res = agent.run_cv(quick_family, bundle, y, screen_folds, fast=screen_fast,
                               entity_mode=mode, residual=residual)
        except Exception as exc:  # noqa: BLE001
            print(f"   [Screen] {name}: ошибка ({exc})")
            res = None
        bundle.entity_enabled = prev_entity
        bundle.hier_enabled = prev_hier
        if res is None:
            return None
        verdict = critic.evaluate(name, res["oof_score"], res.get("fold_scores"))
        return res if verdict["accepted"] else None

    base_res = screen("base (raw features)", entity=False, hier=False)
    if base_res is not None:
        best_state = {"entity": False, "blocks": ["base"]}

    if bundle.entity is not None and remaining() > 300:
        bundle.entity_enabled = True
        res_loo = screen("entity history (LOO train / past eval)", entity=True, mode=MODE_LOO)
        best_mode = MODE_LOO
        if res_loo is not None:
            best_state["entity"] = True
        if remaining() > 300:
            res_before = screen("entity history (expanding only)", entity=True, mode=MODE_BEFORE)
            if res_before is not None:
                best_mode = MODE_BEFORE
                best_state["entity"] = True
        entity_mode = best_mode if best_state["entity"] else MODE_LOO
    else:
        entity_mode = MODE_LOO

    # generic hierarchy block: coarse attribute-group target history
    if bundle.hier and remaining() > 300:
        if "hier" not in bundle.blocks:
            bundle.blocks.append("hier")
        hier_res = screen("+ hierarchical target history", entity=best_state["entity"],
                          mode=entity_mode, hier=True)
        if hier_res is not None:
            best_state["blocks"].append("hier")
            bundle.hier_enabled = True
        else:
            bundle.hier_enabled = False
            bundle.blocks = [b for b in bundle.blocks if b != "hier"]

    if remaining() > 300:
        if bundle.add_numeric_interactions():
            if screen("+ numeric ratios/products", entity=best_state["entity"],
                      mode=entity_mode, hier=("hier" in best_state["blocks"])) is not None:
                best_state["blocks"].append("numx")
            else:
                bundle.drop_block("numx")

    if remaining() > 300:
        if bundle.add_group_relative_features():
            if screen("+ group-relative numerics", entity=best_state["entity"],
                      mode=entity_mode, hier=("hier" in best_state["blocks"])) is not None:
                best_state["blocks"].append("relx")
            else:
                bundle.drop_block("relx")

    if bundle.cat_features and remaining() > 300:
        if bundle.add_frequency_features():
            if screen("+ frequency encoding", entity=best_state["entity"], mode=entity_mode,
                      hier=("hier" in best_state["blocks"])) is not None:
                best_state["blocks"].append("freq")
            else:
                for c in [c for c in bundle.X_tr.columns if c.startswith("freq_")]:
                    bundle.X_tr.drop(columns=[c], inplace=True, errors="ignore")
                    bundle.X_te.drop(columns=[c], inplace=True, errors="ignore")
                bundle.drop_block("freq")

    if bundle.cat_features and remaining() > 300:
        if bundle.add_cross_features():
            if screen("+ categorical crosses", entity=best_state["entity"], mode=entity_mode,
                      hier=("hier" in best_state["blocks"])) is not None:
                best_state["blocks"].append("crosses")
            else:
                bundle.drop_block("crosses")
                bundle.X_tr.drop(columns=[c for c in bundle.X_tr.columns if c.startswith("x_")],
                                 inplace=True, errors="ignore")
                bundle.X_te.drop(columns=[c for c in bundle.X_te.columns if c.startswith("x_")],
                                 inplace=True, errors="ignore")

    residual_mode = False
    if profile.is_regression and best_state["entity"] and remaining() > 400:
        res = screen("entity residual (two-part model)", entity=True, residual=True,
                     mode=entity_mode, hier=("hier" in best_state["blocks"]))
        if res is not None:
            residual_mode = True

    bundle.entity_enabled = best_state["entity"]
    bundle.hier_enabled = "hier" in best_state["blocks"] and bool(bundle.hier)
    print(f"[FeatureArchitect] итоговый набор: {bundle.describe()}; "
          f"entity_mode={entity_mode}, two-part={residual_mode}")

    # ---- PHASE 4: optional LLM feature hypothesis --------------------------
    llm_history: List[str] = []
    if remaining() > 1500 and profile.n_train > 50:
        end = phase("Фаза 4/6 — LLM-гипотезы признаков (ограниченно, 2 итерации)", 900)
        try:
            from task_profiler import _text  # local import: only needed here
            task_text = _text(task_dir)
        except Exception:
            task_text = ""
        generator = LLMFeatureGenerator(profile, train_df, task_text)
        if generator.available:
            for it in range(2):
                if time.time() > end - 60:
                    break
                hypothesis = generator.propose(llm_history)
                if hypothesis is None:
                    break
                cols_name = f"llm{it}_"
                try:
                    train_feats = generator.materialise(hypothesis["code"], train_df)
                    test_feats = generator.materialise(hypothesis["code"], test_df)
                except Exception as exc:  # noqa: BLE001
                    print(f"   [LLMFeatures] итерация {it + 1} не удалась: {exc}")
                    continue
                if train_feats is None or test_feats is None or train_feats.shape[1] == 0:
                    llm_history.append(f"{hypothesis['hypothesis']}: не удалось материализовать")
                    continue
                added = []
                for c in train_feats.columns:
                    new_name = f"{cols_name}{c}"
                    bundle.X_tr[new_name] = train_feats[c].values
                    bundle.X_te[new_name] = test_feats[c].values
                    added.append(new_name)
                decision = screen(f"LLM: {hypothesis['hypothesis'][:60]}",
                                  entity=best_state["entity"], mode=entity_mode,
                                  residual=residual_mode,
                                  hier=("hier" in best_state["blocks"]))
                if decision is None:
                    bundle.X_tr.drop(columns=added, inplace=True, errors="ignore")
                    bundle.X_te.drop(columns=added, inplace=True, errors="ignore")
                    llm_history.append(f"{hypothesis['hypothesis']}: без улучшения")
                else:
                    llm_history.append(f"{hypothesis['hypothesis']}: принято")
        else:
            print("[LLMFeatures] LLM недоступен — фаза пропущена (детерминированные блоки уже сильны).")

    # ---- PHASE 5: full model portfolio -------------------------------------
    phase("Фаза 5/6 — портфель моделей и ансамблирование", max(300.0, remaining() - 600))
    families = agent.model_families()
    for fam in families:
        if remaining() <= 120 and agent.results:
            print(f"[ModelPortfolio] пропуск {fam}: недостаточно времени")
            continue
        if remaining() <= 120 and not agent.results:
            print(f"[ModelPortfolio] времени мало, но портфель пуст — обучаем {fam} (fast)")
            try:
                res = agent.run_cv(fam, bundle, y, folds, fast=True,
                                   entity_mode=entity_mode, residual=residual_mode)
            except Exception as exc:  # noqa: BLE001
                print(f"   ⚠️ {fam}: {exc}")
                res = None
            if res is not None:
                agent.results[fam] = res
                print(f"   → {fam}: {profile.metric}={res['oof_score']:.4f} (fast)")
                critic.evaluate(f"portfolio:{fam}", res["oof_score"], res.get("fold_scores"))
            continue
        try:
            res = agent.run_cv(fam, bundle, y, folds, fast=False,
                               entity_mode=entity_mode, residual=residual_mode)
        except Exception as exc:  # noqa: BLE001
            print(f"   ⚠️ {fam}: {exc}")
            res = None
        if res is None:
            continue
        agent.results[fam] = res
        print(f"   → {fam}: {profile.metric}={res['oof_score']:.4f} "
              f"(фолды: {[round(s, 4) for s in res['fold_scores']]})")
        critic.evaluate(f"portfolio:{fam}", res["oof_score"], res.get("fold_scores"))

    # softmax + one-vs-rest hybrid: OvR fixes intrinsically confusable class pairs
    if profile.is_classification and remaining() > 400:
        try:
            base_res = agent.results.get("lgbm")
            ovr_res = agent.results.get("lgbm:ovr")
            if base_res is not None and ovr_res is not None:
                agent._y_ref = np.asarray(y, dtype=float)
                merged = agent.merge_results(base_res, ovr_res, tag="ovr", w_base=0.5)
                if merged is not base_res:
                    agent.results["lgbm+ovr"] = merged
                    print(f"   → softmax+OvR: {profile.metric}={merged['oof_score']:.4f}")
        except Exception as exc:  # noqa: BLE001
            print(f"[ModelPortfolio] OvR-слияние пропущено: {exc}")

    # multi-seed bagging: average every family over a second seed while time allows
    # (variance reduction with no selection risk)
    if remaining() > 1200 and agent.results:
        agent._y_ref = np.asarray(y, dtype=float)
        try:
            ranked = sorted(agent.results.items(), key=lambda kv: kv[1]["oof_score"],
                            reverse=(critic.direction == "maximize"))
            for fam, base_res in ranked:
                if remaining() < 900:
                    break
                extra = agent.run_cv(fam, bundle, y, folds, fast=False,
                                     entity_mode=entity_mode, residual=residual_mode,
                                     seed=SEED + 1)
                if extra is None:
                    continue
                merged = agent.merge_results(base_res, extra, tag="seed2")
                if merged is not base_res:
                    agent.results[fam] = merged
                    print(f"   → {fam}: seed-бэггинг {profile.metric}={merged['oof_score']:.4f}")
        except Exception as exc:  # noqa: BLE001
            print(f"[ModelPortfolio] seed-бэггинг пропущен: {exc}")

    # third seed for the two strongest families: cheap variance reduction on the blend
    if remaining() > 1800 and agent.results:
        agent._y_ref = np.asarray(y, dtype=float)
        try:
            ranked = sorted(agent.results.items(), key=lambda kv: kv[1]["oof_score"],
                            reverse=(critic.direction == "maximize"))
            for fam, base_res in ranked[:2]:
                if remaining() < 1200:
                    break
                extra = agent.run_cv(fam, bundle, y, folds, fast=False,
                                     entity_mode=entity_mode, residual=residual_mode,
                                     seed=SEED + 2)
                if extra is None:
                    continue
                merged = agent.merge_results(base_res, extra, tag="seed3", w_base=2.0 / 3.0)
                if merged is not base_res:
                    agent.results[fam] = merged
                    print(f"   → {fam}: 3-seed бэггинг {profile.metric}={merged['oof_score']:.4f}")
        except Exception as exc:  # noqa: BLE001
            print(f"[ModelPortfolio] 3-seed бэггинг пропущен: {exc}")

    # last resort: a single cheap model must exist before phase 6
    if not agent.results and remaining() > 40:
        for fam in families[:2]:
            try:
                res = agent.run_cv(fam, bundle, y, folds, fast=True, entity_mode=entity_mode,
                                   residual=residual_mode)
            except Exception:  # noqa: BLE001
                res = None
            if res is not None:
                agent.results[fam] = res
                print(f"   → {fam}: {profile.metric}={res['oof_score']:.4f} (аварийный минимум)")
                break

    # diversity model for regression (cheap, adds ensemble variety)
    if profile.is_regression and remaining() > 240:
        try:
            res = agent.run_cv("extra_trees", bundle, y, folds, fast=False, entity_mode=entity_mode)
            if res is not None:
                agent.results["extra_trees"] = res
                print(f"   → extra_trees: {profile.metric}={res['oof_score']:.4f}")
        except Exception as exc:  # noqa: BLE001
            print(f"   ⚠️ extra_trees: {exc}")

    # ---- PHASE 6: blending + decision rule + final submission --------------
    phase("Фаза 6/6 — ансамбль, калибровка решения и финальный сабмит", max(180.0, remaining() - 30))
    blend = None
    try:
        blend = agent.blend(y)
    except Exception as exc:  # noqa: BLE001
        print(f"[ModelPortfolio] блендинг не удался: {exc}")

    multipliers = None
    if profile.is_classification and blend is not None and blend.get("oof") is not None:
        try:
            mask = blend.get("mask")
            if mask is not None and mask.sum() > 100:
                mask = np.asarray(mask, dtype=bool)
                multipliers = agent.tune_class_multipliers(np.asarray(blend["oof"])[mask], y[mask])
            else:
                multipliers = agent.tune_class_multipliers(np.asarray(blend["oof"]), y)
        except Exception as exc:  # noqa: BLE001
            print(f"[ModelPortfolio] калибровка классов не удалась: {exc}")

    # final test predictions: fold-bagged blend, optionally averaged with a full refit
    test_proba = blend.get("test") if blend is not None else None
    test_pred = None
    if profile.is_classification:
        if test_proba is not None:
            test_proba = np.asarray(test_proba, dtype=float)
            if multipliers is not None:
                test_proba = test_proba * np.asarray(multipliers, dtype=float).reshape(1, -1)
        else:
            _, best_res = _best_result(agent, critic.direction)
            test_proba = best_res.get("test_proba") if best_res else None
    else:
        test_pred = blend.get("test") if blend is not None else None
        if test_pred is None:
            _, best_res = _best_result(agent, critic.direction)
            test_pred = best_res.get("test_pred") if best_res else None
        if test_pred is None:
            test_pred = np.full(len(test_df), float(np.nanmedian(y)))

    # full-data refit of the strongest family, averaged with the bagged prediction
    if remaining() > 420 and agent.results:
        try:
            best_name, _ = _best_result(agent, critic.direction)
            refit = agent.refit_full(best_name, bundle, y, entity_mode=entity_mode,
                                     residual=residual_mode and profile.is_regression)
            if refit is not None:
                if profile.is_classification and refit.get("test_proba") is not None:
                    rp = np.asarray(refit["test_proba"], dtype=float)
                    if multipliers is not None:
                        rp = rp * np.asarray(multipliers, dtype=float).reshape(1, -1)
                    test_proba = 0.5 * test_proba + 0.5 * rp if test_proba is not None else rp
                elif not profile.is_classification and refit.get("test_pred") is not None:
                    rpred = np.asarray(refit["test_pred"], dtype=float)
                    test_pred = 0.5 * np.asarray(test_pred, dtype=float) + 0.5 * rpred
                print(f"[ModelPortfolio] дообучение {best_name} на всём train добавлено в ансамбль (0.5)")
        except Exception as exc:  # noqa: BLE001
            print(f"[ModelPortfolio] дообучение пропущено: {exc}")

    final_pred = _final_predictions(profile, test_pred, test_proba, labels)
    if final_pred is None:
        ok, msg = validate_submission(task_dir, profile, n_expected=len(test_df))
        print(f"\n[Submission] оставлен ранее записанный файл → {msg}")
    else:
        written = save_submission(task_dir, test_ids, final_pred, profile)
        ok, msg = validate_submission(task_dir, profile, n_expected=len(test_df))
        print(f"\n[Submission] {written} → {msg}")

    # ---- report ------------------------------------------------------------
    print("\n" + "=" * 78)
    print("📊 ИТОГОВЫЙ ОТЧЁТ")
    print("=" * 78)
    print(critic.summary())
    if agent.results:
        print("   OOF качество моделей:")
        for name, res in sorted(agent.results.items(), key=lambda kv: -kv[1]["oof_score"]
                                if critic.direction == "maximize" else kv[1]["oof_score"]):
            print(f"     • {name:12s} {profile.metric}={res['oof_score']:.4f}")
    if blend is not None:
        print(f"   ансамбль: {blend['info']} → {profile.metric}={blend['score']:.4f}")
        if blend.get("candidates"):
            print(f"   кандидаты ансамбля: {blend['candidates']}")
    print(f"   время работы: {time.time() - t_start:.0f}s; потоков: {N_JOBS}")
    print("=" * 78)
    return ok


def _best_result(agent, direction: str):
    """Pick the strongest CV result honouring the metric direction."""
    if not agent.results:
        return None, None
    ordered = sorted(agent.results.items(), key=lambda kv: kv[1]["oof_score"],
                     reverse=(direction == "maximize"))
    return ordered[0]


def _final_predictions(profile: TaskProfile, test_pred, test_proba, labels):
    """Convert model outputs into submission values (labels / probabilities / numbers)."""
    if test_pred is None and test_proba is None:
        # nothing was trained in time: keep the already written fail-safe submission
        print("[Submission] нет обученных моделей — сохраняем текущий submission.csv без изменений")
        return None
    if profile.is_classification:
        if profile.predict_proba and test_proba is not None:
            proba = np.asarray(test_proba, dtype=float)
            return proba[:, 1] if proba.shape[1] == 2 else proba
        probs = test_proba
        if probs is not None:
            idx = np.argmax(np.asarray(probs, dtype=float), axis=1)
        else:
            idx = np.asarray(test_pred, dtype=int)
        labels_arr = np.asarray(labels if labels else profile.class_labels, dtype=object)
        if len(labels_arr) == 0:
            return idx
        return labels_arr[np.clip(idx, 0, len(labels_arr) - 1)]
    pred = np.asarray(test_pred, dtype=float)
    if profile.target_transform == "log1p":
        pred = np.expm1(pred)
    lo = profile.target_min
    hi = profile.target_max
    if lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        pad = 0.02 * (hi - lo)
        pred = np.clip(pred, lo - pad, hi + pad)
    finite = np.isfinite(pred)
    if not finite.all():
        fill = float(np.nanmedian(pred[finite])) if finite.any() else 0.0
        pred = np.where(finite, pred, fill)
    return pred


def _write_predictions(task_dir: Path, profile: TaskProfile, test_ids, test_proba, test_pred,
                       labels, agent, multipliers) -> None:
    try:
        if profile.is_classification and test_proba is not None:
            proba = np.asarray(test_proba, dtype=float)
            if multipliers is not None:
                proba = proba * np.asarray(multipliers, dtype=float).reshape(1, -1)
        else:
            proba = None
        final = _final_predictions(profile, test_pred, proba, labels)
        if final is not None:
            save_submission(task_dir, test_ids, final, profile)
    except Exception as exc:  # noqa: BLE001
        print(f"[FailSafe] не удалось записать промежуточный сабмит: {exc}")


def multilabel_pipeline(task_dir, profile: TaskProfile, train_df, test_df, test_ids, phase, remaining):
    """Minimal but valid multilabel path: independent binary models per target column."""
    phase("Мультилейбл-ветка", max(300.0, remaining() - 60))
    from sklearn.ensemble import HistGradientBoostingClassifier
    sub = pd.DataFrame({profile.id_column: np.asarray(test_ids)})
    try:
        bundle = FeatureBundle(profile, train_df, test_df)
        agent = ModelPortfolioAgent(profile, n_jobs=N_JOBS, seed=SEED)
        for col in profile.target_columns:
            yb = pd.to_numeric(train_df[col], errors="coerce").fillna(0).astype(int).values
            X_tr = bundle.frame("train", None, mode=MODE_LOO, include_entity=False)
            X_te = bundle.frame("test", None, mode=MODE_ALL, include_entity=False)
            cats = bundle.cat_features_for(X_tr)
            from model_agent import _ordinal_encode
            Xt, _, Xe = _ordinal_encode(X_tr, X_tr.head(2), cats, X_te)
            model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.07,
                                                   random_state=SEED)
            model.fit(Xt, yb)
            proba = np.asarray(model.predict_proba(Xe), dtype=float)
            sub[col] = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
            print(f"   • {col}: записаны вероятности")
    except Exception as exc:  # noqa: BLE001
        print(f"[Multilabel] ошибка ({exc}); пишем нули")
        for col in profile.target_columns:
            if col not in sub.columns:
                sub[col] = 0.0
    try:
        sub.columns = [profile.id_column] + profile.target_columns
        sub.to_csv(Path(task_dir) / "submission.csv", index=False, encoding="utf-8")
        print(f"[Multilabel] submission.csv записан: {len(sub)} строк")
    except Exception:
        pass
    return True


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    task_argument = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        main(task_argument)
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001 — never exit non-zero
        print("[CRITICAL] Непойманная ошибка пайплайна:\n" + traceback.format_exc())
        try:
            task_md = resolve_task_path(task_argument)
            task_dir = task_md.parent
            if not (task_dir / "submission.csv").exists():
                write_blind_failsafe(task_dir)
        except BaseException:
            pass
    # Hard exit: skips interpreter-shutdown noise (e.g. multiprocessing ResourceTracker
    # tracebacks) that could otherwise be mistaken for a pipeline crash.
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except BaseException:
        pass
    os._exit(0)
