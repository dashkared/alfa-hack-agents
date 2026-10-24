import json
import os
import re
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
from prompts import FEATURE_ARCHITECT_SYSTEM_PROMPT, FeatureProposal, TaskProfile
from task_profiler import build_llm_client
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential


class FeatureArchitectAgent:
    """Агент 2: Инженер признаков. Проектирует доменные формулы и генерирует код по словарю данных."""

    def __init__(self, task_dir: Path, profile: TaskProfile):
        self.task_dir = Path(task_dir)
        self.profile = profile
        self.model = os.environ.get("OPENAI_MODEL", "qwen/qwen3.8-27b")

    def _read_data_dictionary_summary(self) -> str:
        """Считывает словарь данных с русскими описаниями полей (компактная версия)."""
        dict_path = self.task_dir / "data_dictionary.csv"
        if dict_path.exists():
            try:
                df_dict = pd.read_csv(dict_path)
                lines = []
                for _, row in df_dict.iterrows():
                    feat = row.get("feature", "")
                    desc = row.get("description_ru", "")
                    role = row.get("role", "")
                    if feat and feat != self.profile.target_column and feat != self.profile.id_column:
                        lines.append(f"- `{feat}`: {desc} (тип: {role})")
                return "\n".join(lines[:25])
            except Exception:
                pass
        return "Словарь данных отсутствует или не прочитан."

    def _apply_domain_fallback_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Детерминированный доменный бейзлайн фичей, если LLM недоступен или код упал."""
        df = df.copy()

        # Домен 1: Сетевые атаки (UNSW-NB15)
        if "sbytes" in df.columns and "dbytes" in df.columns:
            sbytes = pd.to_numeric(df["sbytes"], errors="coerce").fillna(0).astype(np.float32)
            dbytes = pd.to_numeric(df["dbytes"], errors="coerce").fillna(0).astype(np.float32)
            spkts = pd.to_numeric(df["spkts"], errors="coerce").fillna(1).astype(np.float32) if "spkts" in df.columns else 1.0
            dpkts = pd.to_numeric(df["dpkts"], errors="coerce").fillna(1).astype(np.float32) if "dpkts" in df.columns else 1.0
            dur = pd.to_numeric(df["dur"], errors="coerce").fillna(0.001).astype(np.float32) if "dur" in df.columns else 0.001

            # Робастное логарифмирование для защиты от выбросов
            log_sbytes = np.log1p(sbytes.clip(lower=0))
            log_dbytes = np.log1p(dbytes.clip(lower=0))
            log_spkts = np.log1p(spkts.clip(lower=0))
            log_dpkts = np.log1p(dpkts.clip(lower=0))

            df["fe_byte_ratio_log"] = log_sbytes - log_dbytes
            df["fe_pkt_ratio_log"] = log_spkts - log_dpkts
            df["fe_sbyte_rate_log"] = log_sbytes / (dur + 0.001)
            df["fe_dbyte_rate_log"] = log_dbytes / (dur + 0.001)
            df["fe_spkt_mean_size"] = log_sbytes / (spkts + 1.0)
            df["fe_dpkt_mean_size"] = log_dbytes / (dpkts + 1.0)

            if "sttl" in df.columns and "dttl" in df.columns:
                sttl = pd.to_numeric(df["sttl"], errors="coerce").fillna(0).astype(np.float32)
                dttl = pd.to_numeric(df["dttl"], errors="coerce").fillna(0).astype(np.float32)
                df["fe_ttl_diff"] = sttl - dttl

            if "synack" in df.columns and "ackdat" in df.columns:
                synack = pd.to_numeric(df["synack"], errors="coerce").fillna(0).astype(np.float32)
                ackdat = pd.to_numeric(df["ackdat"], errors="coerce").fillna(0).astype(np.float32)
                df["fe_tcp_setup_time"] = synack + ackdat

            # Точечные признаки для сложных классов (Analysis, Backdoor, DoS)
            if "ct_src_dport_ltm" in df.columns and "ct_dst_sport_ltm" in df.columns:
                c1 = pd.to_numeric(df["ct_src_dport_ltm"], errors="coerce").fillna(0).astype(np.float32)
                c2 = pd.to_numeric(df["ct_dst_sport_ltm"], errors="coerce").fillna(0).astype(np.float32)
                df["fe_port_scan_ratio"] = (c1 + 1.0) / (c2 + 1.0)

            if "ct_dst_src_ltm" in df.columns and "ct_srv_dst" in df.columns:
                c3 = pd.to_numeric(df["ct_dst_src_ltm"], errors="coerce").fillna(0).astype(np.float32)
                c4 = pd.to_numeric(df["ct_srv_dst"], errors="coerce").fillna(0).astype(np.float32)
                df["fe_dst_src_ratio"] = (c3 + 1.0) / (c4 + 1.0)

            if "swin" in df.columns and "dwin" in df.columns:
                swin = pd.to_numeric(df["swin"], errors="coerce").fillna(0).astype(np.float32)
                dwin = pd.to_numeric(df["dwin"], errors="coerce").fillna(0).astype(np.float32)
                df["fe_win_diff"] = swin - dwin

            if "rate" in df.columns:
                rate = pd.to_numeric(df["rate"], errors="coerce").fillna(0).astype(np.float32)
                df["fe_rate_log"] = np.log1p(rate.clip(lower=0))

        # Домен 2: Энергопотребление зданий (Регрессия)
        if "cooling_degree_days_c" in df.columns and "heating_degree_days_c" in df.columns:
            cdd = pd.to_numeric(df["cooling_degree_days_c"], errors="coerce").fillna(0).astype(np.float32)
            hdd = pd.to_numeric(df["heating_degree_days_c"], errors="coerce").fillna(0).astype(np.float32)
            df["fe_degree_days_ratio"] = (cdd + 1.0) / (hdd + 1.0)
            df["fe_total_degree_days"] = cdd + hdd

            months = [
                "january", "february", "march", "april", "may", "june",
                "july", "august", "september", "october", "november", "december"
            ]
            for m in months:
                max_c = f"{m}_max_temp_c"
                min_c = f"{m}_min_temp_c"
                if max_c in df.columns and min_c in df.columns:
                    mx = pd.to_numeric(df[max_c], errors="coerce").fillna(0).astype(np.float32)
                    mn = pd.to_numeric(df[min_c], errors="coerce").fillna(0).astype(np.float32)
                    df[f"fe_{m}_temp_spread"] = mx - mn

            if "Year_Factor" in df.columns and "year_built" in df.columns:
                yf = pd.to_numeric(df["Year_Factor"], errors="coerce").fillna(0).astype(np.float32)
                yb = pd.to_numeric(df["year_built"], errors="coerce").replace(0, np.nan).fillna(1970).astype(np.float32)
                df["fe_building_age_proxy"] = (yf * 10 - yb)

        return df

    def apply_universal_tabular_features(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        cat_features: List[str],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """Фундаментальный генератор табличных фичей: даты, индикаторы пропусков, частоты, сглаженные групповые агрегаты."""
        print("[FeatureArchitect] Генерация фундаментальных табличных признаков (Universal Tabular Engine)...")
        X_tr = X_train.copy()
        X_te = X_test.copy()
        new_cats = list(cat_features)

        # 1. Автоматическое распознавание дат и извлечение календарных признаков
        for col in X_train.columns:
            if col in cat_features or X_train[col].dtype == object:
                sample_s = X_train[col].dropna().astype(str).head(30)
                if not sample_s.empty and sample_s.str.contains(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}").mean() > 0.5:
                    try:
                        tr_dt = pd.to_datetime(X_tr[col], errors="coerce")
                        te_dt = pd.to_datetime(X_te[col], errors="coerce")
                        if tr_dt.notna().mean() > 0.5:
                            X_tr[f"fe_{col}_year"] = tr_dt.dt.year.astype(np.float32)
                            X_tr[f"fe_{col}_month"] = tr_dt.dt.month.astype(np.float32)
                            X_tr[f"fe_{col}_day"] = tr_dt.dt.day.astype(np.float32)
                            X_tr[f"fe_{col}_dow"] = tr_dt.dt.dayofweek.astype(np.float32)
                            X_tr[f"fe_{col}_is_weekend"] = (tr_dt.dt.dayofweek >= 5).astype(np.float32)

                            X_te[f"fe_{col}_year"] = te_dt.dt.year.astype(np.float32)
                            X_te[f"fe_{col}_month"] = te_dt.dt.month.astype(np.float32)
                            X_te[f"fe_{col}_day"] = te_dt.dt.day.astype(np.float32)
                            X_te[f"fe_{col}_dow"] = te_dt.dt.dayofweek.astype(np.float32)
                            X_te[f"fe_{col}_is_weekend"] = (te_dt.dt.dayofweek >= 5).astype(np.float32)

                            if (tr_dt.dt.hour != 0).any():
                                X_tr[f"fe_{col}_hour"] = tr_dt.dt.hour.astype(np.float32)
                                X_tr[f"fe_{col}_hour_sin"] = np.sin(2 * np.pi * tr_dt.dt.hour / 24.0).astype(np.float32)
                                X_tr[f"fe_{col}_hour_cos"] = np.cos(2 * np.pi * tr_dt.dt.hour / 24.0).astype(np.float32)

                                X_te[f"fe_{col}_hour"] = te_dt.dt.hour.astype(np.float32)
                                X_te[f"fe_{col}_hour_sin"] = np.sin(2 * np.pi * te_dt.dt.hour / 24.0).astype(np.float32)
                                X_te[f"fe_{col}_hour_cos"] = np.cos(2 * np.pi * te_dt.dt.hour / 24.0).astype(np.float32)
                    except Exception:
                        pass

        # 2. Индикаторы пропусков (Missingness Flags) для признаков с пропусками > 3%
        for col in X_train.columns:
            na_ratio = float(X_train[col].isna().mean())
            if na_ratio > 0.03:
                col_name = f"fe_{col}_isna"
                X_tr[col_name] = X_train[col].isna().astype(np.float32)
                X_te[col_name] = X_test[col].isna().astype(np.float32)

        # 3. Частотное кодирование (Frequency Encoding) для всех категориальных признаков
        for col in cat_features:
            if col in X_tr.columns:
                freq = X_tr[col].value_counts(normalize=True)
                freq_col = f"fe_{col}_freq"
                X_tr[freq_col] = X_tr[col].map(freq).fillna(0.0).astype(np.float32)
                X_te[freq_col] = X_te[col].map(freq).fillna(0.0).astype(np.float32)

        # 4. Сглаженные групповые агрегации (Bayesian Shrinkage Groupby Statistics: mean & diff)
        # Находим категориальные колонки с кардинальностью от 2 до 80
        cand_cats = [c for c in cat_features if c in X_tr.columns and 2 <= X_tr[c].nunique() <= 80]
        priority_cats = ["facility_type", "building_class", "State_Factor", "proto", "service", "state"]
        sorted_cats = [c for c in priority_cats if c in cand_cats] + [c for c in cand_cats if c not in priority_cats]
        active_cats = sorted_cats[:3]

        # Находим числовые колонки с высокой дисперсией
        num_cols = [c for c in X_tr.columns if pd.api.types.is_numeric_dtype(X_tr[c]) and not c.startswith("fe_")]
        priority_nums = [
            "floor_area", "energy_star_rating", "year_built", "cooling_degree_days_c", "heating_degree_days_c",
            "sbytes", "dbytes", "sttl", "dttl", "dur", "spkts", "dpkts"
        ]
        sorted_nums = [c for c in priority_nums if c in num_cols] + [c for c in num_cols if c not in priority_nums]
        active_nums = sorted_nums[:6]

        for cat in active_cats:
            for num in active_nums:
                try:
                    num_series = pd.to_numeric(X_tr[num], errors="coerce")
                    global_mean = float(num_series.dropna().mean()) if not num_series.dropna().empty else 0.0
                    grp_stats = num_series.groupby(X_tr[cat]).agg(["mean", "count"])
                    smoothing = 10.0
                    smoothed_mean = (grp_stats["mean"] * grp_stats["count"] + smoothing * global_mean) / (grp_stats["count"] + smoothing)
                    mean_col = f"fe_{num}_grp_{cat}_mean"
                    diff_col = f"fe_{num}_diff_{cat}"

                    X_tr[mean_col] = X_tr[cat].map(smoothed_mean).fillna(global_mean).astype(np.float32)
                    X_te[mean_col] = X_te[cat].map(smoothed_mean).fillna(global_mean).astype(np.float32)

                    X_tr[diff_col] = (pd.to_numeric(X_tr[num], errors="coerce").fillna(global_mean) - X_tr[mean_col]).astype(np.float32)
                    X_te[diff_col] = (pd.to_numeric(X_te[num], errors="coerce").fillna(global_mean) - X_te[mean_col]).astype(np.float32)
                except Exception:
                    pass

        # 5. Отношения ключевых числовых величин
        if len(active_nums) >= 2:
            n1, n2 = active_nums[0], active_nums[1]
            try:
                s1 = pd.to_numeric(X_tr[n1], errors="coerce").fillna(0).astype(np.float32)
                s2 = pd.to_numeric(X_tr[n2], errors="coerce").fillna(0).astype(np.float32)
                t1 = pd.to_numeric(X_te[n1], errors="coerce").fillna(0).astype(np.float32)
                t2 = pd.to_numeric(X_te[n2], errors="coerce").fillna(0).astype(np.float32)
                X_tr[f"fe_{n1}_div_{n2}"] = s1 / (s2.abs() + 1.0)
                X_te[f"fe_{n1}_div_{n2}"] = t1 / (t2.abs() + 1.0)
            except Exception:
                pass

        # 6. Попарные категориальные кроссы (2-Way Categorical Crosses + Frequency)
        if len(active_cats) >= 2:
            for i in range(min(len(active_cats), 3)):
                for j in range(i + 1, min(len(active_cats), 3)):
                    c1, c2 = active_cats[i], active_cats[j]
                    try:
                        if X_tr[c1].nunique() <= 70 and X_tr[c2].nunique() <= 70:
                            cross_col = f"fe_{c1}_x_{c2}"
                            tr_cross = X_tr[c1].astype(str) + "_" + X_tr[c2].astype(str)
                            te_cross = X_te[c1].astype(str) + "_" + X_te[c2].astype(str)
                            X_tr[cross_col] = tr_cross
                            X_te[cross_col] = te_cross
                            if cross_col not in new_cats:
                                new_cats.append(cross_col)

                            # Совместная частота кросс-категории
                            cr_freq = tr_cross.value_counts(normalize=True)
                            X_tr[f"{cross_col}_freq"] = tr_cross.map(cr_freq).fillna(0.0).astype(np.float32)
                            X_te[f"{cross_col}_freq"] = te_cross.map(cr_freq).fillna(0.0).astype(np.float32)
                    except Exception:
                        pass

        # 7. Доменные эвристики для известных модальностей
        X_tr = self._apply_domain_fallback_features(X_tr)
        X_te = self._apply_domain_fallback_features(X_te)

        added_cols = [c for c in X_tr.columns if c not in X_train.columns]
        print(f"[FeatureArchitect] Фундаментальные фичи созданы ({len(added_cols)} новых колонок): {added_cols[:6]}")
        return X_tr, X_te, new_cats

    def _extract_code(self, raw_code: str) -> str:
        """Извлечение чистого Python-кода из markdown блоков."""
        if "```python" in raw_code:
            match = re.search(r"```python\s*(.*?)\s*```", raw_code, re.DOTALL)
            if match:
                return match.group(1)
        elif "```" in raw_code:
            match = re.search(r"```\s*(.*?)\s*```", raw_code, re.DOTALL)
            if match:
                return match.group(1)
        return raw_code

    def generate_feature_transform(
        self,
        df_sample: pd.DataFrame,
        critique_feedback: str = "",
    ) -> Tuple[Callable[[pd.DataFrame], pd.DataFrame], str]:
        """Генерирует функцию создания признаков через Qwen с защитой и fallback."""
        client = build_llm_client()
        dict_text = self._read_data_dictionary_summary()

        if not client:
            print("[FeatureArchitect] LLM API недоступен, применяются доменные преобразования признаков.")
            return self._apply_domain_fallback_features, "Domain heuristic feature engineering pipeline"

        sample_cols = list(df_sample.columns)[:25]
        prompt = f"""
Контекст ML-задачи:
- Тип задачи: {self.profile.task_type}
- Целевая колонка: {self.profile.target_column}
- Метрика: {self.profile.metric} ({self.profile.objective})

Словарь ключевых колонок:
{dict_text}

Доступные колонки в датасете:
{sample_cols}

Замечания критика с прошлой итерации:
{critique_feedback if critique_feedback else "Это первая итерация. Сгенерируй 3-5 ключевых доменных признаков."}

ВАЖНО: Рассуждай кратко (не более 2 предложений). Сразу выводи JSON с функцией add_features(df).
ОБЯЗАТЕЛЬНЫЕ ПРАВИЛА:
1. Для величин с широким диапазоном (байты, пакеты, длительность, площади) ОБЯЗАТЕЛЬНО используй np.log1p(pd.to_numeric(df[col], errors='coerce').clip(lower=0)) перед делением или вычитанием!
2. Не предлагай тривиальные разности существующих колонок, создавай нелинейные комбинации (разности логарифмов, пороговые бинарные флаги, нормализованные интенсивности).
3. Используй pd.to_numeric(df[col], errors='coerce') для всех числовых колонок!
"""
        import openai

        @retry(
            wait=wait_random_exponential(multiplier=1, max=20),
            stop=stop_after_attempt(2),
            retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError)),
            reraise=True,
        )
        def call_llm():
            print(f"[FeatureArchitect] Запрос к LLM ({self.model}) для генерации доменных фичей...")
            return client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": FEATURE_ARCHITECT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=6500,
            )

        try:
            resp = call_llm()
            msg = resp.choices[0].message
            content = msg.content
            if not content and hasattr(msg, "reasoning_content") and msg.reasoning_content:
                content = msg.reasoning_content
            elif not content and hasattr(msg, "provider_specific_fields") and msg.provider_specific_fields:
                psf = msg.provider_specific_fields
                content = psf.get("reasoning") or psf.get("reasoning_content") or ""

            if not content:
                content = str(resp)

            match = re.search(r"\{.*\}", content, re.DOTALL)
            raw_json = match.group(0) if match else content
            data = None

            # 1. Попытка стандартного парсинга
            try:
                data = json.loads(raw_json, strict=False)
            except Exception:
                try:
                    fixed_json = re.sub(r'(?<!\\)\n', r'\\n', raw_json)
                    data = json.loads(fixed_json, strict=False)
                except Exception:
                    pass

            # 2. Если JSON распарсился — извлекаем FeatureProposal
            if data and isinstance(data, dict):
                proposal = FeatureProposal(**data)
                code = self._extract_code(proposal.python_code)
            else:
                # 3. Регулярное выражение для извлечения def add_features напрямую
                code_match = re.search(r"(def add_features\(.*?\):\n(?:[ \t]+.*?\n)+)", content)
                if code_match:
                    code = code_match.group(1)
                    proposal = FeatureProposal(
                        hypotheses=[FeatureHypothesis(feature_name="llm_extracted_feat", rationale="Прямое извлечение функции из ответа LLM")],
                        python_code=code,
                    )
                    print("[FeatureArchitect] Функция add_features успешно извлечена напрямую из текста модели в обход поврежденного JSON.")
                else:
                    raise ValueError("Не удалось распарсить JSON и не найдена функция add_features")

            print(f"[FeatureArchitect] Получено {len(proposal.hypotheses)} гипотез признаков:")
            for h in proposal.hypotheses:
                print(f"  • {h.feature_name}: {h.rationale}")

            # Песочница: безопасная компиляция и тестовый прогон функции
            local_scope = {"pd": pd, "np": np}
            exec(code, local_scope)

            if "add_features" in local_scope and callable(local_scope["add_features"]):
                transform_fn = local_scope["add_features"]
                test_out = transform_fn(df_sample.head(20))
                if isinstance(test_out, pd.DataFrame) and len(test_out) == 20:
                    new_cols = set(test_out.columns) - set(df_sample.columns)
                    print(f"[FeatureArchitect] Тест в песочнице успешен! Создано {len(new_cols)} новых признаков: {list(new_cols)[:6]}")
                    return transform_fn, f"LLM Feature Proposal ({len(new_cols)} features)"

        except Exception as e:
            print(f"[FeatureArchitect] Ошибка генерации или исполнения кода фичей ({e}). Переключение на доменный fallback.")

        return self._apply_domain_fallback_features, "Domain heuristic fallback features"
