"""Прогон демо: синтетический набор -> те же проверки и прогноз, что в основном приложении.

Модель, проверка входа и прогноз — существующие функции пакета fraud_app (без изменений и без копий):
load_booster (SHA-256, число деревьев, имена признаков), read_csv_text + prepare (проверка 458 признаков,
категории как при обучении), predict_proba (booster.predict), result_table (класс = вероятность >= порога).
Порог — значение по умолчанию из config/app_config.json (0.04861665045870177).
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_app.io_csv import read_csv_text
from fraud_app.model_loader import ModelError, load_booster, model_categories
from fraud_app.predict import predict_proba, result_table
from fraud_app.validation import prepare

from . import synthetic

RESULT_SUSPICIOUS, RESULT_LOW = "Подозрительная", "Низкий риск"


@dataclass
class DemoResult:
    data: pd.DataFrame            # синтетический набор: TransactionID + 458 признаков
    results: pd.DataFrame         # result_table основного приложения (в порядке набора)
    threshold: float
    n_total: int
    n_suspicious: int
    n_low: int
    n_unknown_categories: int
    warnings: list


def load_model(model_path, cfg, schema):
    """Только локальный файл модели (MODEL_PATH): демо не обращается к облаку."""
    if not model_path:
        raise ModelError("не задана переменная окружения MODEL_PATH — путь к файлу модели lgbm_B_full.txt")
    p = Path(model_path)
    if not p.is_file():
        raise ModelError(f"файл модели не найден: {p}")
    return load_booster(p, cfg, schema)


def run_demo(booster, cfg, schema, params, spec, n=synthetic.N_DEMO, seed=synthetic.SEED) -> DemoResult:
    cats = model_categories(booster, schema)
    data = synthetic.generate(schema, cats, params, spec, n=n, seed=seed)
    prep = prepare(read_csv_text(synthetic.to_csv_bytes(data)), schema, cats, cfg["max_rows"])
    if prep.errors:
        raise synthetic.DemoDataError("синтетический набор не прошёл проверку приложения: " + "; ".join(prep.errors))
    threshold = cfg["default_threshold"]
    p = predict_proba(booster, prep.X)
    res = result_table(p, threshold, prep.ids, schema.id_column)
    n_sus = int(res["predicted_class"].sum())
    n_unk = 0 if prep.unknown_categories is None else int(prep.unknown_categories["строк с незнакомым значением"].sum())
    return DemoResult(data, res, threshold, len(res), n_sus, len(res) - n_sus, n_unk, list(prep.warnings))


def display_table(res: pd.DataFrame, id_column: str) -> pd.DataFrame:
    """Таблица для экрана: от самого высокого риска к самому низкому, русские названия столбцов."""
    order = np.argsort(-res["fraud_probability"].to_numpy(), kind="stable")
    r = res.iloc[order]
    # «Результат проверки» — вторым столбцом: на узком экране он виден без прокрутки вбок
    return pd.DataFrame({
        "Номер операции": r[id_column].to_numpy(),
        "Результат проверки": np.where(r["predicted_class"].to_numpy() == 1,
                                       "🔴 " + RESULT_SUSPICIOUS, "🟢 " + RESULT_LOW),
        "Вероятность мошенничества": r["fraud_probability"].to_numpy() * 100.0,
        "Уровень риска": r["fraud_probability"].to_numpy(),
    })


def risk_bands(p: np.ndarray, threshold: float) -> pd.DataFrame:
    """Число операций по диапазонам вероятности (диапазоны только для графика; граница класса — сам порог)."""
    thr_pct = f"{threshold * 100:.2f}".replace(".", ",")
    edges = [0.0, 1e-4, 1e-3, 1e-2, threshold, 0.10, 0.50, np.inf]
    labels = ["< 0,01%", "0,01–0,1%", "0,1–1%", f"1–{thr_pct}%", f"{thr_pct}–10%", "10–50%", "≥ 50%"]
    counts = pd.cut(pd.Series(p), edges, labels=labels, right=False).value_counts().reindex(labels, fill_value=0)
    return pd.DataFrame({
        "Диапазон вероятности": labels,
        "Операций": counts.to_numpy().astype(int),
        "Результат": [RESULT_LOW] * 4 + [RESULT_SUSPICIOUS] * 3,
        "Порядок": range(len(labels)),
    })
