"""Сквозная проверка интерфейса через streamlit.testing (без браузера). Нужна модель (MODEL_PATH)."""
import io

import pandas as pd
from streamlit.testing.v1 import AppTest

from conftest import ROOT

APP = str(ROOT / "app.py")


def _app(booster):  # booster — чтобы тест пропускался без MODEL_PATH
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception, at.exception
    return at


def test_starts_and_loads_model(booster):
    at = _app(booster)
    assert any("Модель проверена" in c.value for c in at.caption)
    assert len(at.file_uploader) == 1 and not at.error


def test_missing_feature_shows_error(booster, schema):
    at = _app(booster)
    cols = [n for n in schema.names if n != "C1"]
    data = pd.DataFrame([[""] * len(cols)], columns=cols).to_csv(index=False).encode("utf-8")
    at.file_uploader[0].set_value(("bad.csv", data, "text/csv"))
    at.run()
    assert not at.exception
    assert any("C1" in e.value for e in at.error) and len(at.metric) == 0


def test_full_flow_local(booster, cfg, local_csv):
    # Ожидаемые значения не хранятся в коде: они считаются из локального эталона (вне репозитория).
    ref = pd.read_csv(local_csv / "validation_2000_reference_p_LGB2.csv")["p_LGB2"].to_numpy()
    expected = {t: int((ref >= t).sum()) for t in (cfg["default_threshold"], 0.5)}
    at = _app(booster)
    data = (local_csv / "validation_2000_features.csv").read_bytes()
    at.file_uploader[0].set_value(("validation.csv", data, "text/csv"))
    at.run()
    assert not at.exception and not at.error
    m = {x.label: x.value for x in at.metric}
    assert m["Операций"] == f"{len(ref):,}"
    assert m[f"Тревог (класс 1) при пороге {cfg['default_threshold']:.4f}"] == f"{expected[cfg['default_threshold']]:,}"
    assert any(f"Фактический порог классификации: {cfg['default_threshold']:.10g}" in c.value for c in at.caption)
    at.number_input[0].set_value(0.5)
    at.run()
    m = {x.label: x.value for x in at.metric}
    assert m["Тревог (класс 1) при пороге 0.5000"] == f"{expected[0.5]:,}"
    assert any("Фактический порог классификации: 0.5 " in c.value for c in at.caption)
    assert any("Скачать результаты" in b.proto.label for b in at.get("download_button"))


def test_results_file_name():
    from fraud_app.predict import results_file_name
    sha = "f413c46fe5a22e1873af666562459702eb94a9dda00126a73b9d27809e757f2c"
    assert results_file_name(sha, 0.04861665045870177) == "lgbm2_results_f413c46f_thr0.0486.csv"
    assert results_file_name(sha, 0.5) == "lgbm2_results_f413c46f_thr0.5000.csv"
