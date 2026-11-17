"""ТОЛЬКО ЛОКАЛЬНО. Подготовка CSV признаков (458) для выборки строк Validation — для приёмочного теста приложения.

Запуск — в окружении исследовательского проекта (нужны pyarrow, psutil и данные проекта; в этот репозиторий они не входят):
    python tools/export_features_local.py <папка_вывода_вне_репозитория> [n_rows] --project-root <корень_проекта>
Корень проекта можно задать и переменной окружения PROJECT_ROOT. В корне ожидаются data/processed/*, models/comparison_09/*,
models/controlled_13/lgbm_B_full.txt и reports/final_m4_vs_lgbm2/validation_predictions.parquet.

Код сборки признаков — без изменений из ноутбука 14 (build_B, состояние T). Validation используется только для чтения;
ничего не обучается и не подбирается. Результат НЕ публикуется: реальные данные соревнования (правила Kaggle).
Перед записью CSV проверяется, что прогноз LightGBM II на собранной таблице совпадает с сохранённым прогнозом ноутбука 14.
"""
import argparse
import gc
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq

GB = 1024 ** 3
MIN_AVAIL = [psutil.virtual_memory().available / GB]


def guard():
    while True:
        a = psutil.virtual_memory().available / GB
        MIN_AVAIL[0] = min(MIN_AVAIL[0], a)
        if a < 1.0:
            print(f"ОСТАНОВКА: свободно {a:.2f} ГБ < 1 ГБ", flush=True)
            os._exit(3)
        time.sleep(0.25)


threading.Thread(target=guard, daemon=True).start()
import lightgbm as lgb  # noqa: E402

ap = argparse.ArgumentParser(description="Локальный экспорт признаков Validation для приёмочного теста (не публикуется)")
ap.add_argument("out_dir", help="папка вывода вне репозитория приложения")
ap.add_argument("n_rows", nargs="?", type=int, default=2000, help="число случайных строк Validation (seed 42)")
ap.add_argument("--project-root", default=os.environ.get("PROJECT_ROOT"),
                help="корень исследовательского проекта (или переменная окружения PROJECT_ROOT)")
args = ap.parse_args()
if not args.project_root:
    ap.error("не задан корень проекта: --project-root или переменная окружения PROJECT_ROOT")
OUT = Path(args.out_dir).resolve()
N_ROWS = args.n_rows
APP = Path(__file__).resolve().parent.parent
assert not str(OUT).lower().startswith(str(APP).lower()), "папка вывода должна быть вне репозитория приложения"
ROOT = Path(args.project_root).resolve()
if not (ROOT / "data" / "processed").is_dir():
    ap.error(f"в корне проекта нет data/processed: {ROOT}")
OUT.mkdir(parents=True, exist_ok=True)
PROCESSED, C09, C13 = ROOT / "data" / "processed", ROOT / "models" / "comparison_09", ROOT / "models" / "controlled_13"
TP_, IP_ = PROCESSED / "train_transaction.parquet", PROCESSED / "train_identity.parquet"
feature_sets = json.loads((C09 / "feature_sets.json").read_text(encoding="utf-8"))
spec = json.loads((PROCESSED / "lr_feature_spec.json").read_text(encoding="utf-8"))
TRANS_COLS, IDENT_COLS = pq.read_schema(TP_).names, pq.read_schema(IP_).names

key = pd.read_parquet(TP_, columns=["TransactionID", "TransactionDT"])
ORDER = np.argsort(key["TransactionDT"].to_numpy(), kind="stable")
dt_all = key["TransactionDT"].to_numpy()[ORDER]
N = len(dt_all)
cutoff = dt_all[int(N * 0.8)]
n_tr = int((dt_all < cutoff).sum())
ids_va = key["TransactionID"].to_numpy()[ORDER][n_tr:]
# Строки Validation сверяются с сохранённым результатом ноутбука 14 (TransactionID и порядок), без констант из данных в коде
REF_PATH = ROOT / "reports" / "final_m4_vs_lgbm2" / "validation_predictions.parquet"
ref_ids = pd.read_parquet(REF_PATH, columns=["TransactionID"])["TransactionID"].to_numpy()
assert len(ref_ids) == len(ids_va) and np.array_equal(ref_ids.astype("int64"), ids_va.astype("int64")), \
    "строки Validation не совпадают с ноутбуком 14"
del ref_ids
del key


def cols_with_prefix(prefix):
    return [c for c in TRANS_COLS + IDENT_COLS if c.startswith(prefix) and c[len(prefix):].isdigit()]


FEATURES_03 = ([c for c in TRANS_COLS if c not in ("isFraud", "TransactionID", "TransactionDT")]
               + [c for c in IDENT_COLS if c != "TransactionID"]
               + ["has_identity", "hour", "weekday", "n_missing_V", "n_missing_D", "n_missing_M", "n_missing_id"])
A_01_2 = ["amount_group", "is_large_amount", "card_type_group", "card_category_group", "P_email_provider", "R_email_provider",
          "P_email_domain_freq", "R_email_domain_freq", "email_same_domain", "device_os", "device_brand", "device_model_family"]
A_01_3 = ["group_prev_transactions", "group_prev_amount_mean", "group_prev_amount_min", "group_prev_amount_max", "group_time_since_prev"]
A_08 = ["group_first_operation", "group_time_since_prev_bin", "group_prev_amount_mean_bin"]
FEAT_B = FEATURES_03 + A_01_2 + A_01_3 + A_08
assert len(FEAT_B) == 458 and FEAT_B == feature_sets["set_B"]
EDGES = lambda v: [np.inf if e == "inf" else float(e) for e in v]  # noqa: E731
H = spec["history_from_01_3"]


def build_B(pos_sorted):
    # без изменений из ноутбука 14
    file_rows = ORDER[np.asarray(pos_sorted)]
    parts = []
    for i in range(0, len(TRANS_COLS), 40):
        ch = pd.read_parquet(TP_, columns=TRANS_COLS[i:i + 40])
        parts.append(ch.iloc[file_rows].reset_index(drop=True))
        del ch
        gc.collect()
    df = pd.concat(parts, axis=1)
    del parts
    ident = pd.read_parquet(IP_)
    has_identity = df["TransactionID"].isin(ident["TransactionID"]).astype("int8")
    df = df.merge(ident, on="TransactionID", how="left", validate="one_to_one")
    del ident
    df["has_identity"] = has_identity.values
    f12 = pd.read_parquet(PROCESSED / "interpretable_features.parquet")
    f13 = pd.read_parquet(PROCESSED / "group_history_features.parquet")
    df = (df.merge(f12.drop(columns="split"), on="TransactionID", how="left", validate="one_to_one")
            .merge(f13.rename(columns={"split": "split_01_3"}), on="TransactionID", how="left", validate="one_to_one"))
    del f12, f13
    gc.collect()
    assert np.all(np.diff(df["TransactionDT"].to_numpy()) >= 0)
    df["hour"] = ((df["TransactionDT"] // 3600) % 24).astype("int8")
    df["weekday"] = ((df["TransactionDT"] // 86_400) % 7).astype("int8")
    df["n_missing_V"] = df[cols_with_prefix("V")].isna().sum(axis=1).astype("int16")
    df["n_missing_D"] = df[cols_with_prefix("D")].isna().sum(axis=1).astype("int8")
    df["n_missing_M"] = df[cols_with_prefix("M")].isna().sum(axis=1).astype("int8")
    df["n_missing_id"] = df[cols_with_prefix("id_")].isna().sum(axis=1).astype("int8")
    df["TransactionAmt"] = df["TransactionAmt"].astype("float32")
    cnt = df["group_prev_transactions"]
    has_hist = cnt.fillna(0) > 0
    df["group_first_operation"] = (cnt == 0).astype("int8")
    df["group_time_since_prev_bin"] = (pd.cut(df["group_time_since_prev"], EDGES(H["group_time_since_prev_bin"]["edges_sec"]),
                                              labels=H["group_time_since_prev_bin"]["labels"], right=False).astype(object).where(has_hist, np.nan))
    df["group_prev_amount_mean_bin"] = (pd.cut(df["group_prev_amount_mean"], EDGES(H["group_prev_amount_mean_bin"]["edges_usd"]),
                                               labels=H["group_prev_amount_mean_bin"]["labels"], right=False).astype(object).where(has_hist, np.nan))
    ids = df["TransactionID"].to_numpy()
    X = df[FEAT_B].copy()
    del df
    gc.collect()
    cat_cols = []
    for c in FEAT_B:
        s = X[c]
        if isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == object or pd.api.types.is_string_dtype(s.dtype):
            X[c] = s.astype("category")
            cat_cols.append(c)
    assert cat_cols == feature_sets["lgbm_categorical"] and len(cat_cols) == 41
    return X, ids, cat_cols


rng = np.random.default_rng(42)
pos = np.sort(rng.choice(np.arange(n_tr, N), size=N_ROWS, replace=False))
X, ids, CAT = build_B(pos)
print(f"собрано {len(X):,} строк × {X.shape[1]}; процесс {psutil.Process().memory_info().rss / GB:.2f} ГБ")

# 1) таблица = таблица ноутбука 14: прогноз совпадает с сохранённым p_LGB2
booster = lgb.Booster(model_file=str(C13 / "lgbm_B_full.txt"))
p = booster.predict(X)
ref = pd.read_parquet(REF_PATH, columns=["TransactionID", "p_LGB2"])
j = pd.DataFrame({"TransactionID": ids, "p": p}).merge(ref, on="TransactionID", how="inner", validate="one_to_one")
assert len(j) == N_ROWS
d_frame = float(np.abs(j["p"] - j["p_LGB2"]).max())
print(f"проверка таблицы: max |прогноз − p_LGB2 ноутбука 14| = {d_frame:.3e}")
assert d_frame <= 1e-12, "собранная таблица не воспроизводит прогноз ноутбука 14 — остановка"

# 2) запись CSV: числа — float64 с точным обратным чтением (repr), пропуск — пустая ячейка, категории — текст как есть
cols = {"TransactionID": ids}
for c in FEAT_B:
    if c in CAT:
        v = X[c].astype(object)
        cols[c] = v.where(v.notna(), "").to_numpy()
    else:
        cols[c] = X[c].astype("float64").to_numpy()
out = pd.DataFrame(cols)
csv_path = OUT / f"validation_{N_ROWS}_features.csv"
out.to_csv(csv_path, index=False, encoding="utf-8", na_rep="", lineterminator="\n")
j[["TransactionID", "p_LGB2"]].to_csv(OUT / f"validation_{N_ROWS}_reference_p_LGB2.csv", index=False, encoding="utf-8",
                                      float_format="%.17g")
print(f"записано: {csv_path} ({csv_path.stat().st_size / 1024 ** 2:.1f} МБ)")
print(f"пик процесса: {getattr(psutil.Process().memory_info(), 'peak_wset', float('nan')) / GB:.2f} ГБ; "
      f"мин. свободно в системе: {MIN_AVAIL[0]:.2f} ГБ")
