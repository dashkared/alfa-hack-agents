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
        """Считывает словарь данных с русскими описаниями полей."""
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
                return "\n".join(lines[:60])
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

            df["fe_byte_ratio"] = (sbytes + 1.0) / (dbytes + 1.0)
            df["fe_pkt_ratio"] = (spkts + 1.0) / (dpkts + 1.0)
            df["fe_sbyte_rate"] = sbytes / (dur + 0.001)
            df["fe_dbyte_rate"] = dbytes / (dur + 0.001)
            df["fe_spkt_mean_size"] = sbytes / (spkts + 1.0)
            df["fe_dpkt_mean_size"] = dbytes / (dpkts + 1.0)

            if "sttl" in df.columns and "dttl" in df.columns:
                sttl = pd.to_numeric(df["sttl"], errors="coerce").fillna(0).astype(np.float32)
                dttl = pd.to_numeric(df["dttl"], errors="coerce").fillna(0).astype(np.float32)
                df["fe_ttl_diff"] = sttl - dttl

            if "synack" in df.columns and "ackdat" in df.columns:
                synack = pd.to_numeric(df["synack"], errors="coerce").fillna(0).astype(np.float32)
                ackdat = pd.to_numeric(df["ackdat"], errors="coerce").fillna(0).astype(np.float32)
                df["fe_tcp_setup_time"] = synack + ackdat

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

        sample_cols = list(df_sample.columns)[:40]
        prompt = f"""
Контекст ML-задачи:
- Тип задачи: {self.profile.task_type}
- Целевая колонка: {self.profile.target_column}
- Метрика: {self.profile.metric} ({self.profile.objective})

Словарь признаков (смысл колонок):
{dict_text}

Доступные колонки в датасете:
{sample_cols}

Замечания критика с прошлой итерации:
{critique_feedback if critique_feedback else "Это первая итерация. Сгенерируй 4-6 ключевых доменных признаков."}

Напиши компактную и надежную функцию add_features(df).
ОБЯЗАТЕЛЬНО используй pd.to_numeric(df[col], errors='coerce') перед сложением/вычитанием/делением, чтобы избежать ошибок с типами object!
"""
        import openai

        @retry(
            wait=wait_random_exponential(multiplier=1, max=30),
            stop=stop_after_attempt(3),
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
                temperature=0.2,
                max_tokens=3500,
            )

        try:
            resp = call_llm()
            msg = resp.choices[0].message
            content = msg.content
            if not content and hasattr(msg, "reasoning_content") and msg.reasoning_content:
                content = msg.reasoning_content

            if not content:
                raise ValueError("LLM response content is empty")

            match = re.search(r"\{.*\}", content, re.DOTALL)
            raw_json = match.group(0) if match else content
            data = None

            # 1. Попытка стандартного парсинга с разрешением управляющих символов
            try:
                data = json.loads(raw_json, strict=False)
            except Exception:
                # 2. Попытка исправить неэкранированные переносы строк внутри строковых литералов
                try:
                    fixed_json = re.sub(r'(?<!\\)\n', r'\\n', raw_json)
                    data = json.loads(fixed_json, strict=False)
                except Exception:
                    pass

            # 3. Если JSON распарсился — создаем FeatureProposal
            if data and isinstance(data, dict):
                proposal = FeatureProposal(**data)
                code = self._extract_code(proposal.python_code)
            else:
                # 4. Если JSON поврежден (например, неэкранированные кавычки в коде),
                # вытаскиваем функцию def add_features напрямую через регулярное выражение!
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
                # Тестируем на 20 строках сэмпла
                test_out = transform_fn(df_sample.head(20))
                if isinstance(test_out, pd.DataFrame) and len(test_out) == 20:
                    new_cols = set(test_out.columns) - set(df_sample.columns)
                    print(f"[FeatureArchitect] Тест в песочнице успешен! Создано {len(new_cols)} новых признаков: {list(new_cols)[:6]}")
                    return transform_fn, f"LLM Feature Proposal ({len(new_cols)} features)"

        except Exception as e:
            print(f"[FeatureArchitect] Ошибка генерации или исполнения кода фичей ({e}). Переключение на доменный fallback.")

        return self._apply_domain_fallback_features, "Domain heuristic fallback features"
