"""Приёмочный тест — ТОЛЬКО ЛОКАЛЬНО: прогноз приложения по CSV = сохранённый прогноз ноутбука 14 (p_LGB2) для тех же
TransactionID, допуск 1e-12. Реальные строки Validation и эталон лежат вне репозитория (LOCAL_DATA_DIR)."""
import numpy as np
import pandas as pd

from fraud_app.io_csv import read_csv_text
from fraud_app.model_loader import model_categories
from fraud_app.predict import predict_proba, result_table
from fraud_app.validation import prepare

TOL = 1e-12


def _compare(booster, schema, cfg, data, ref):
    cats = model_categories(booster, schema)
    r = prepare(read_csv_text(data), schema, cats, cfg["max_rows"])
    assert not r.errors, r.errors
    p = predict_proba(booster, r.X)
    out = result_table(p, cfg["default_threshold"], r.ids, schema.id_column)
    out["TransactionID"] = out["TransactionID"].astype("int64")
    j = out.merge(ref, on="TransactionID", how="inner", validate="one_to_one")
    assert len(j) == len(ref) == len(out)
    return float(np.max(np.abs(j["fraud_probability"] - j["p_LGB2"]))), r, out


def test_acceptance_same_predictions(booster, schema, cfg, local_csv):
    data = (local_csv / "validation_2000_features.csv").read_bytes()
    ref = pd.read_csv(local_csv / "validation_2000_reference_p_LGB2.csv")
    d, r, out = _compare(booster, schema, cfg, data, ref)
    print(f"\nприёмка: строк {len(out)}, max |Δ| = {d:.3e}, тревог при 0.0486: {int(out['predicted_class'].sum())}, "
          f"признаков с незнакомыми категориями: {len(r.unknown_categories)}")
    assert d <= TOL


def test_acceptance_reordered_columns(booster, schema, cfg, local_csv):
    df = pd.read_csv(local_csv / "validation_2000_features.csv", dtype=str, keep_default_na=False)
    rng = np.random.default_rng(7)
    cols = ["TransactionID"] + list(rng.permutation(list(schema.names)))
    data = df[cols].to_csv(index=False, lineterminator="\n").encode("utf-8")
    ref = pd.read_csv(local_csv / "validation_2000_reference_p_LGB2.csv")
    d, r, _ = _compare(booster, schema, cfg, data, ref)
    assert any("переставлены" in w for w in r.warnings) and d <= TOL
