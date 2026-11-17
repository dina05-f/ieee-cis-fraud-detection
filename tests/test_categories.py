"""Категориальные признаки: соответствие фактическому поведению LightGBM II (ноутбуки 13–14).

Нужна настоящая модель (MODEL_PATH). Для части проверок — локальные реальные строки (LOCAL_DATA_DIR);
без них используются синтетические строки. Реальные данные в репозиторий не попадают.
"""
import numpy as np
import pandas as pd
import pytest

from fraud_app.io_csv import read_csv_text
from fraud_app.model_loader import model_categories
from fraud_app.validation import prepare


def manual_matrix(X, schema, cats):
    """Независимое кодирование: код категории = позиция значения в словаре модели; незнакомое / пропуск -> NaN."""
    A = np.empty(X.shape, dtype=np.float64)
    for j, c in enumerate(schema.names):
        if c in cats:
            pos = {v: i for i, v in enumerate(cats[c])}
            A[:, j] = [pos.get(v, np.nan) if isinstance(v, str) else np.nan for v in X[c].astype(object)]
        else:
            A[:, j] = X[c].to_numpy(dtype=np.float64)
    return A


def base_rows(schema, cats, local_csv_dir=None, n=200):
    if local_csv_dir is not None:
        data = (local_csv_dir / "validation_2000_features.csv").read_bytes()
        r = prepare(read_csv_text(data), schema, cats, 50_000)
        assert not r.errors
        return r.X.iloc[:n].reset_index(drop=True)
    rng = np.random.default_rng(1)
    rows = {}
    for c in schema.names:
        if c in cats:
            vals = cats[c] + [np.nan]
            rows[c] = [vals[k] for k in rng.integers(0, len(vals), n)]
        else:
            rows[c] = np.where(rng.random(n) < 0.3, np.nan, rng.normal(size=n) * 100)
    X = pd.DataFrame(rows)[list(schema.names)]
    for c in schema.categorical:
        X[c] = X[c].astype(object).astype("category")
    return X


def test_model_dictionaries(booster, schema):
    cats = model_categories(booster, schema)
    assert list(cats) == list(schema.categorical) and len(cats) == 41
    assert all(isinstance(v, str) for lst in cats.values() for v in lst)
    NA_TOKENS = {"", "NA", "N/A", "NaN", "nan", "None", "null", "NULL", "<NA>", "n/a", "#N/A"}
    assert not any(v in NA_TOKENS for lst in cats.values() for v in lst)
    assert all(v == v.strip() for lst in cats.values() for v in lst)
    # не-ASCII значения, которые должны прочитаться из CSV без искажений
    assert "≥1000" in cats["amount_group"] and "50–100" in cats["amount_group"]


@pytest.mark.parametrize("use_local", [False, True])
def test_dataframe_path_equals_independent_encoding(booster, schema, use_local, request):
    cats = model_categories(booster, schema)
    loc = request.getfixturevalue("local_csv") if use_local else None
    X = base_rows(schema, cats, loc)
    p_df = booster.predict(X)                                 # путь приложения и ноутбуков 13–14
    p_np = booster.predict(manual_matrix(X, schema, cats))    # независимое кодирование
    assert np.max(np.abs(p_df - p_np)) <= 1e-12


@pytest.mark.parametrize("token", ["NA", "None", "–", "≥", "null", "НЕИЗВЕСТНО"])
def test_unknown_category_equals_missing(booster, schema, token):
    """Незнакомое значение категории обрабатывается моделью как пропуск — для каждого из 41 признака."""
    cats = model_categories(booster, schema)
    X = base_rows(schema, cats, n=50)
    for c in schema.categorical:
        assert token not in cats[c]
        Xu, Xn = X.copy(), X.copy()
        Xu[c] = pd.Series([token] * len(X), dtype=object).astype("category")
        Xn[c] = pd.Series([np.nan] * len(X), dtype=object).astype("category")
        assert np.max(np.abs(booster.predict(Xu) - booster.predict(Xn))) <= 1e-12, c


def test_csv_reading_keeps_known_values(booster, schema):
    """Известные значения (в том числе с «–», «≥», кириллицей) после CSV получают тот же код, что в словаре модели."""
    cats = model_categories(booster, schema)
    n = max(len(v) for v in cats.values())
    data = {}
    for c in schema.names:
        if c in cats:
            data[c] = [cats[c][i % len(cats[c])] for i in range(n)]
        else:
            data[c] = [""] * n
    data["amount_group"][0] = ""                             # пустая ячейка -> пропуск
    raw = pd.DataFrame(data).to_csv(index=False, lineterminator="\n").encode("utf-8")
    r = prepare(read_csv_text(raw), schema, cats, 50_000)
    assert not r.errors
    for c in schema.categorical:
        exp = [cats[c][i % len(cats[c])] for i in range(n)]
        got = r.X[c].astype(object).tolist()
        if c == "amount_group":
            assert pd.isna(got[0]) and got[1:] == exp[1:]
        else:
            assert got == exp, c
    assert r.unknown_categories.empty
    p_df = booster.predict(r.X)
    p_np = booster.predict(manual_matrix(r.X, schema, cats))
    assert np.max(np.abs(p_df - p_np)) <= 1e-12
