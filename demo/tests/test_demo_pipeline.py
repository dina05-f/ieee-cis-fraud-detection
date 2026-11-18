"""Прогнозы демо — настоящая LightGBM II через функции основного приложения; порог 0.0486 не меняется."""
import hashlib
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np

from fraud_app.io_csv import read_csv_text
from fraud_app.model_loader import model_categories
from fraud_app.validation import prepare
from fraud_demo import pipeline, synthetic


def test_model_is_the_real_lightgbm_ii(booster, cfg, schema):
    assert isinstance(booster, lgb.Booster)
    assert booster.num_trees() == cfg["n_trees"] == 4628
    assert tuple(booster.feature_name()) == schema.names and len(schema.names) == 458
    h = hashlib.sha256(Path(os.environ["MODEL_PATH"]).read_bytes()).hexdigest()
    assert h == cfg["model_sha256"]


def test_threshold_unchanged(demo, cfg):
    assert demo.threshold == cfg["default_threshold"] == 0.04861665045870177
    assert (demo.results["threshold"] == cfg["default_threshold"]).all()


def test_predictions_equal_direct_booster_predict(demo, booster, cfg, schema):
    # независимый прогон: та же проверка входа, затем booster.predict напрямую
    cats = model_categories(booster, schema)
    prep = prepare(read_csv_text(synthetic.to_csv_bytes(demo.data)), schema, cats, cfg["max_rows"])
    assert not prep.errors
    direct = booster.predict(prep.X)
    assert np.array_equal(demo.results["fraud_probability"].to_numpy(), direct)
    p = demo.results["fraud_probability"].to_numpy()
    assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()


def test_classes_and_counts(demo):
    p = demo.results["fraud_probability"].to_numpy()
    cls = demo.results["predicted_class"].to_numpy()
    assert (cls == (p >= demo.threshold)).all()
    assert demo.n_total == len(p) == demo.n_suspicious + demo.n_low
    assert demo.n_suspicious == int((p >= demo.threshold).sum())
    assert demo.n_unknown_categories == 0 and demo.warnings == []
    assert demo.n_suspicious > 0 and demo.n_low > 0                               # оба результата показаны


def test_display_table_sorted_and_russian(demo, schema):
    t = pipeline.display_table(demo.results, schema.id_column)
    assert list(t.columns) == ["Номер операции", "Результат проверки", "Вероятность мошенничества", "Уровень риска"]
    assert (np.diff(t["Уровень риска"].to_numpy()) <= 0).all()                    # от высокого риска к низкому
    assert np.allclose(t["Вероятность мошенничества"], t["Уровень риска"] * 100)
    sus = t["Результат проверки"].str.contains(pipeline.RESULT_SUSPICIOUS)
    assert (sus == (t["Уровень риска"] >= demo.threshold)).all()
    assert len(t) == demo.n_total and int(sus.sum()) == demo.n_suspicious


def test_chart_bands_in_ascending_order(demo):
    from fraud_demo import charts
    b = pipeline.risk_bands(demo.results["fraud_probability"].to_numpy(), demo.threshold)
    spec = charts.risk_chart(b, demo.threshold).to_dict()
    for layer in spec["layer"]:
        assert layer["encoding"]["x"]["sort"] == b["Диапазон вероятности"].tolist()   # от < 0,01% до ≥ 50%
    assert b["Диапазон вероятности"].iloc[0] == "< 0,01%" and b["Диапазон вероятности"].iloc[-1] == "≥ 50%"


def test_risk_bands_split_exactly_at_threshold(demo):
    p = demo.results["fraud_probability"].to_numpy()
    b = pipeline.risk_bands(p, demo.threshold)
    assert b["Операций"].sum() == len(p)
    assert b.loc[b["Результат"] == pipeline.RESULT_SUSPICIOUS, "Операций"].sum() == demo.n_suspicious
