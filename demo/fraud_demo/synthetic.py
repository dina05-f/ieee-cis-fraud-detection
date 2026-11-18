"""Синтетический набор операций для демо (TransactionID + 458 признаков в порядке модели).

Что синтетическое и откуда берутся значения:
- строки IEEE-CIS / Validation / Test НЕ читаются и не копируются; исходные поля генерируются случайно
  с фиксированным seed (набор воспроизводим);
- числовые исходные поля — произвольные синтетические диапазоны (правил не требуют);
- категориальные исходные поля — ТОЛЬКО значения из словарей категорий самой модели:
    * словари до LOW_CARDINALITY_MAX значений (ProductCD, card4/6, M1–M9, id_12…id_38, DeviceType) — любые значения;
    * почтовые домены — только частые: не менее MIN_DOMAIN_COUNT операций в Train (по частотам из параметров 01_2);
    * остальные большие словари (DeviceInfo, id_30, id_31, id_33) — частоты значений неизвестны, поэтому
      редкие значения не выбираются: поле остаётся пустым (допустимый пропуск);
- производные признаки (has_identity, hour, weekday, n_missing_*, 12 признаков 01_2, история card1+addr1 и 3 производных)
  считаются по существующим правилам проекта (feature_rules.py) — никаких новых правил.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import feature_rules as fr

SEED = 20261118
N_DEMO = 500
LOW_CARDINALITY_MAX = 20
TRAIN_ROWS = 472_432          # строк Train, на которых подобраны параметры 01_2 (аудит preprocessing)
MIN_DOMAIN_COUNT = 1000       # тот же порог «частого» домена, что min_count в параметрах 01_2
PARAM_FILES = ("interpretable_features_params.json", "lr_feature_spec.json")
H5 = ["group_prev_transactions", "group_prev_amount_mean", "group_prev_amount_min", "group_prev_amount_max",
      "group_time_since_prev"]


class DemoDataError(RuntimeError):
    pass


def load_params(params_dir):
    """Параметры 01_2 и границы интервалов истории — локальные приватные файлы проекта (в репозиторий не входят)."""
    if not params_dir:
        raise DemoDataError("не задана переменная окружения FRAUD_PARAMS_DIR — папка с параметрами признаков проекта")
    d = Path(params_dir)
    missing = [f for f in PARAM_FILES if not (d / f).is_file()]
    if missing:
        raise DemoDataError(f"в папке FRAUD_PARAMS_DIR нет файлов: {missing}")
    return tuple(json.loads((d / f).read_text(encoding="utf-8")) for f in PARAM_FILES)


def feature_groups(schema):
    names = list(schema.names)
    raw = names[:names.index("has_identity")]                       # 431 исходный признак
    identity = [c for c in raw if c.startswith("id_") or c in ("DeviceType", "DeviceInfo")]
    if len(raw) != 431 or len(identity) != 40:
        raise DemoDataError("схема признаков не совпадает с ожидаемой (431 исходный, 40 identity)")
    return raw, identity


def allowed_categories(schema, cats, params):
    """Какие значения можно выбирать для каждого исходного категориального признака ([] — только пусто)."""
    raw, _ = feature_groups(schema)
    allowed = {}
    for c in (c for c in raw if c in set(schema.categorical)):
        vocab = list(cats[c])
        if len(vocab) <= LOW_CARDINALITY_MAX:
            allowed[c] = vocab
        elif c in ("P_emaildomain", "R_emaildomain"):
            freq = params["email"]["domain_freq"][c]
            allowed[c] = [v for v in vocab if freq.get(v, 0.0) * TRAIN_ROWS >= MIN_DOMAIN_COUNT]
        else:
            allowed[c] = []
    return allowed


def _numeric(c, rng, n):
    if c == "TransactionAmt":
        return np.round(np.exp(rng.uniform(np.log(1.0), np.log(2000.0), n)), 2)
    if c == "card1":
        return rng.integers(1000, 18000, n).astype(float)
    if c in ("card2", "card3", "card5", "addr1"):
        return rng.integers(100, 600, n).astype(float)
    if c == "addr2":
        return rng.integers(10, 100, n).astype(float)
    if c == "id_02":
        return rng.integers(1, 999_999, n).astype(float)
    if c.startswith("id_"):
        return rng.integers(-100, 101, n).astype(float)
    if c.startswith("V"):
        return rng.integers(0, 4, n).astype(float)
    if c.startswith("C"):
        return rng.integers(0, 15, n).astype(float)
    return rng.integers(0, 600, n).astype(float)                           # D*, dist*


def generate(schema, cats, params, spec, n=N_DEMO, seed=SEED) -> pd.DataFrame:
    """Таблица: TransactionID (DEMO-0001…) + 458 признаков в порядке модели. Детерминирована при заданном seed."""
    rng = np.random.default_rng(seed)
    raw_cols, identity = feature_groups(schema)
    allowed = allowed_categories(schema, cats, params)
    cols = {}
    for c in raw_cols:
        if c in allowed:
            if allowed[c]:
                vals = rng.choice(np.array(allowed[c], dtype=object), n)
                miss = rng.random(n) < 0.2
                cols[c] = pd.Series(np.where(miss, None, vals), dtype=object)
            else:
                cols[c] = pd.Series([None] * n, dtype=object)
        else:
            v = _numeric(c, rng, n)
            miss = np.zeros(n, bool) if c in ("TransactionAmt", "card1") else rng.random(n) < 0.25
            cols[c] = pd.Series(np.where(miss, np.nan, v))
    df = pd.DataFrame(cols)
    has_identity = (rng.random(n) < 0.5).astype("int64")
    df.loc[has_identity == 0, identity] = np.nan                           # нет строки identity -> все 40 пусты

    # история card1+addr1 (смысл 01_3): нет addr1 -> нет ключа; первая операция -> 0 и остальные пусты
    keyed = df["addr1"].notna().to_numpy()
    first = keyed & (rng.random(n) < 0.2)
    withh = keyed & ~first
    n_prev = rng.integers(1, 51, n).astype(float)
    mn = np.round(np.exp(rng.uniform(0.0, np.log(700.0), n)), 2)
    mx = np.round(mn * np.exp(rng.uniform(0.0, np.log(5.0), n)), 2)
    mean = rng.uniform(mn, mx)
    one = n_prev == 1
    mx[one], mean[one] = mn[one], mn[one]
    t = np.round(np.exp(rng.uniform(0.0, np.log(1e7), n)))
    hist = pd.DataFrame(np.nan, index=df.index, columns=H5)
    hist.loc[first, "group_prev_transactions"] = 0.0
    hist.loc[withh, H5] = np.column_stack([n_prev, mean, mn, mx, t])[withh]

    work = pd.concat([df, hist], axis=1)
    work["TransactionDT"] = rng.integers(86_400, 15_000_000, n)            # синтетическое время; в признаки не входит
    work["has_identity"] = has_identity
    f13 = fr.row_features(work, raw_cols)
    f12 = pd.concat([fr.transform_amount(work, params["amount"]), fr.transform_card(work),
                     fr.transform_email(work, params["email"]), fr.transform_device(work, params["device"])], axis=1)
    f08 = fr.history_derived(work, spec)
    full = pd.concat([work.drop(columns="TransactionDT"), f13, f12, f08], axis=1)[list(schema.names)].copy()
    for c in full.columns[full.dtypes == "float32"]:                        # полная точность значения float32
        full[c] = full[c].astype("float64")
    full.insert(0, schema.id_column, [f"DEMO-{i:04d}" for i in range(1, n + 1)])
    return full


def to_csv_bytes(full: pd.DataFrame) -> bytes:
    return full.to_csv(index=False, lineterminator="\n", encoding="utf-8").encode("utf-8")
