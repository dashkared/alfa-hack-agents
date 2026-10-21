# Классификация типов сетевых атак UNSW-NB15

## Задача

Постройте модель многоклассовой классификации, которая по характеристикам
сетевого потока определяет тип атаки. Одна строка соответствует одному
зафиксированному сетевому flow. Целевая переменная `attack_cat` принимает
одно из десяти значений: `Normal`, `Generic`, `Exploits`, `Fuzzers`, `DoS`,
`Reconnaissance`, `Analysis`, `Backdoor`, `Shellcode`, `Worms`.

Классы существенно несбалансированы: `Normal` и `Generic` встречаются
значительно чаще, чем `Worms`, `Shellcode` и `Analysis`. Поэтому основной
метрикой выбран Macro F1, а не Accuracy.

## Данные

Пути указаны относительно каталога этого Markdown-файла:

- `multiclass/participants/train.csv` — 87 670 строк с признаками и таргетом;
- `multiclass/participants/test.csv` — 41 166 строк с признаками без таргета;
- `multiclass/participants/data_dictionary.csv` — описание всех полей;
- `multiclass/participants/sample_submission.csv` — шаблон ответа; значения в нём не являются правильными ответами.

В `train.csv` 44 столбца: 42 входных признака, идентификатор `id` и таргет
`attack_cat`. В `test.csv` 43 столбца: те же признаки и `id`, без таргета.
CSV используют UTF-8, запятую как разделитель и точку как десятичный разделитель.

Test — stratified-выборка 50% из официального hold-out split UNSW-NB15.
Колонка `label` (бинарная метка normal/attack) в данных отсутствует,
поскольку является утечкой для задачи определения конкретного типа атаки.

## Группы признаков

- базовые параметры потока: `dur`, `proto`, `service`, `state`;
- объём трафика: `spkts`, `dpkts`, `sbytes`, `dbytes`, `rate`;
- TTL и нагрузка: `sttl`, `dttl`, `sload`, `dload`, `sloss`, `dloss`;
- временные характеристики: `sinpkt`, `dinpkt`, `sjit`, `djit`, `tcprtt`,
  `synack`, `ackdat`;
- TCP-параметры: `swin`, `dwin`, `stcpb`, `dtcpb`;
- статистика пакетов: `smean`, `dmean`, `trans_depth`, `response_body_len`;
- контекстные счётчики: `ct_srv_src`, `ct_state_ttl`, `ct_dst_ltm`,
  `ct_src_dport_ltm`, `ct_dst_sport_ltm`, `ct_dst_src_ltm`, `ct_src_ltm`,
  `ct_srv_dst`;
- FTP/HTTP: `is_ftp_login`, `ct_ftp_cmd`, `ct_flw_http_mthd`;
- прочее: `is_sm_ips_ports`.

Категориальные признаки: `proto`, `service`, `state`. В них могут
встречаться пропуски; значение `-` означает отсутствие сервиса.
Полное описание находится в `data_dictionary.csv`.

## Что предсказать

Для каждого `id` из `test.csv` необходимо предсказать класс `attack_cat`
(одна из десяти строковых меток, перечисленных выше).

Основная метрика — **Macro F1**:

```python
f1_score(y_true, y_pred, average="macro")
```

Дополнительно рассчитываются **Accuracy** и **Weighted F1**. Для всех метрик
большее значение лучше. Macro F1 выбрана основной, потому что Accuracy
и Weighted F1 не отражают качество на редких классах атак.

## Формат результата

Файл `submission.csv` в каталоге этого Markdown-файла: CSV в UTF-8,
разделитель — запятая, без дополнительного индекса, ровно два столбца:

```csv
id,attack_cat
```

Для каждого `id` из `test.csv` требуется одна строковая метка класса.
Пропуски, повторяющиеся `id`, значения вне списка допустимых классов
и изменение порядка строк недопустимы.
