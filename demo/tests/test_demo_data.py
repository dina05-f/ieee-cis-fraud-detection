"""Синтетический набор демо: воспроизводимость, схема 458 признаков, допустимые категории, правила производных признаков."""
import numpy as np
import pandas as pd

from fraud_app.model_loader import model_categories
from fraud_demo import feature_rules as fr
from fraud_demo import synthetic


def _gen(booster, schema, params, **kw):
    return synthetic.generate(schema, model_categories(booster, schema), *params, **kw)


def test_reproducible_with_fixed_seed(booster, schema, params):
    a, b = _gen(booster, schema, params), _gen(booster, schema, params)
    assert synthetic.to_csv_bytes(a) == synthetic.to_csv_bytes(b)
    c = _gen(booster, schema, params, seed=synthetic.SEED + 1)
    assert synthetic.to_csv_bytes(a) != synthetic.to_csv_bytes(c)


def test_schema_458_features_in_model_order(booster, schema, params):
    d = _gen(booster, schema, params)
    assert list(d.columns) == [schema.id_column] + list(schema.names)
    assert len(d) == synthetic.N_DEMO and d.shape[1] == 459 and "isFraud" not in d.columns
    assert d[schema.id_column].tolist() == [f"DEMO-{i:04d}" for i in range(1, synthetic.N_DEMO + 1)]
    num = d[list(schema.numeric)]
    assert all(pd.api.types.is_numeric_dtype(num[c]) for c in num.columns)
    assert np.isfinite(num.to_numpy(dtype=float)[~np.isnan(num.to_numpy(dtype=float))]).all()


def test_categories_only_from_model_dictionaries(booster, schema, params):
    cats = model_categories(booster, schema)
    d = _gen(booster, schema, params)
    for c in schema.categorical:
        s = d[c].dropna()
        assert s.isin(cats[c]).all(), f"{c}: значения вне словаря модели"
    # большие словари без известных частот — не используются (поле пустое)
    for c in ("DeviceInfo", "id_30", "id_31", "id_33"):
        assert d[c].isna().all(), c
    # почтовые домены — только частые (не менее 1000 операций в Train)
    for c in ("P_emaildomain", "R_emaildomain"):
        freq = params[0]["email"]["domain_freq"][c]
        assert all(freq[v] * synthetic.TRAIN_ROWS >= synthetic.MIN_DOMAIN_COUNT for v in d[c].dropna().unique()), c


def test_derived_features_follow_project_rules(booster, schema, params):
    p01_2, spec = params
    d = _gen(booster, schema, params)
    raw, identity = synthetic.feature_groups(schema)
    hi = d["has_identity"]
    assert set(hi.unique()) <= {0, 1}
    assert d.loc[hi == 0, identity].isna().all().all()                           # нет identity -> 40 полей пусты
    for p in ("V", "D", "M", "id_"):
        cols = fr.digit_cols(raw, p)
        assert (d[f"n_missing_{p.rstrip('_')}"] == d[cols].isna().sum(axis=1)).all(), p
    assert len(fr.digit_cols(raw, "D")) == 15 and len(fr.digit_cols(raw, "id_")) == 38
    assert d["hour"].between(0, 23).all() and d["weekday"].between(0, 6).all()
    f12 = pd.concat([fr.transform_amount(d, p01_2["amount"]), fr.transform_card(d),
                     fr.transform_email(d, p01_2["email"]), fr.transform_device(d, p01_2["device"])], axis=1)
    for c in f12.columns:
        a, b = f12[c], d[c]
        if c in schema.categorical:
            assert (a.astype(object).fillna("<NA>") == b.astype(object).fillna("<NA>")).all(), c
        else:
            assert np.allclose(a.astype(float), b.astype(float), rtol=0, atol=0, equal_nan=True), c
    h = d["group_prev_transactions"]
    H5 = synthetic.H5
    assert d.loc[d["addr1"].isna(), H5].isna().all().all()                        # нет ключа -> история пуста
    assert d.loc[h == 0, H5[1:]].isna().all().all()                               # первая операция группы
    w = d[h > 0]
    assert ((w.group_prev_amount_min <= w.group_prev_amount_mean) & (w.group_prev_amount_mean <= w.group_prev_amount_max)
            & (w.group_time_since_prev > 0)).all()
    f08 = fr.history_derived(d, spec)
    assert (f08["group_first_operation"] == d["group_first_operation"]).all()
    for c in ("group_time_since_prev_bin", "group_prev_amount_mean_bin"):
        assert (f08[c].astype(object).fillna("<NA>") == d[c].astype(object).fillna("<NA>")).all(), c
