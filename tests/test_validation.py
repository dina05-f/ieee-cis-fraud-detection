"""Проверки чтения и валидации на синтетических таблицах (без модели и без реальных данных)."""
import numpy as np
import pandas as pd

from fraud_app.io_csv import read_csv_text
from fraud_app.validation import prepare


def synth_categories(schema):
    return {c: ["a", "b", "NA-не-пропуск", "–", "≥1000"] for c in schema.categorical}


def make_csv(schema, rows=2, with_id=True, override=None, columns=None):
    cols = list(columns) if columns is not None else ((["TransactionID"] if with_id else []) + list(schema.names))
    data = {}
    for c in cols:
        if c == "TransactionID":
            data[c] = [str(1000 + i) for i in range(rows)]
        elif c in schema.categorical:
            data[c] = ["a"] * rows
        else:
            data[c] = ["1.5"] * rows
    df = pd.DataFrame(data, columns=cols)
    for (c, i), v in (override or {}).items():
        df.loc[i, c] = v
    return df.to_csv(index=False, lineterminator="\n").encode("utf-8")


def run(schema, data, max_rows=50_000):
    return prepare(read_csv_text(data), schema, synth_categories(schema), max_rows)


def test_ok(schema):
    r = run(schema, make_csv(schema))
    assert not r.errors and r.X.shape == (2, 458)
    assert tuple(r.X.columns) == schema.names


def test_missing_feature_is_error_not_zero(schema):
    cols = ["TransactionID"] + [n for n in schema.names if n != "C1"]
    r = run(schema, make_csv(schema, columns=cols))
    assert r.X is None and any("C1" in e for e in r.errors)


def test_extra_and_isfraud_rejected(schema):
    cols = ["TransactionID"] + list(schema.names) + ["isFraud"]
    r = run(schema, make_csv(schema, columns=cols))
    assert r.X is None and any("isFraud" in e for e in r.errors)


def test_duplicate_column_rejected(schema):
    cols = ["TransactionID"] + list(schema.names) + ["C1"]
    r = run(schema, make_csv(schema, columns=cols))
    assert r.X is None and any("повторяющиеся" in e for e in r.errors)


def test_reorder_by_name(schema):
    rng = np.random.default_rng(0)
    cols = ["TransactionID"] + list(rng.permutation(schema.names))
    r = run(schema, make_csv(schema, columns=cols))
    assert not r.errors and tuple(r.X.columns) == schema.names and any("переставлены" in w for w in r.warnings)


def test_numeric_text_is_error(schema):
    for bad in ["NA", "nan", "None", "inf", "abc", "1,5"]:
        r = run(schema, make_csv(schema, override={("C1", 0): bad}))
        assert r.X is None and any("C1" in e for e in r.errors), bad


def test_numeric_empty_is_nan(schema):
    r = run(schema, make_csv(schema, override={("C1", 1): ""}))
    assert not r.errors and np.isnan(r.X.loc[1, "C1"]) and r.X.loc[0, "C1"] == 1.5


def test_categorical_text_kept_empty_is_nan(schema):
    c = "amount_group"
    over = {(c, 0): "NA", (c, 1): ""}
    r = run(schema, make_csv(schema, override=over))
    assert not r.errors
    assert r.X.loc[0, c] == "NA"                 # текст «NA» не превращается в пропуск
    assert pd.isna(r.X.loc[1, c])                # пустая ячейка — пропуск
    for v in ["None", "null", "nan", "–", "≥1000", "< 1 мин", " a "]:
        r = run(schema, make_csv(schema, override={(c, 0): v}))
        assert r.X.loc[0, c] == v, v


def test_unknown_categories_reported_not_replaced(schema):
    r = run(schema, make_csv(schema, override={("ProductCD", 0): "НЕИЗВЕСТНО"}))
    assert not r.errors and r.X.loc[0, "ProductCD"] == "НЕИЗВЕСТНО"
    u = r.unknown_categories.set_index("признак")
    assert u.loc["ProductCD", "строк с незнакомым значением"] == 1


def test_not_utf8(schema):
    data = make_csv(schema, override={("group_time_since_prev_bin", 0): "< 1 мин"}).decode("utf-8").encode("cp1251")
    r = run(schema, data)
    assert r.X is None and any("UTF-8" in e for e in r.errors)


def test_row_limit_and_empty(schema):
    assert any("лимита" in e for e in run(schema, make_csv(schema, rows=3), max_rows=2).errors)
    assert any("нет строк" in e for e in run(schema, make_csv(schema, rows=0)).errors)


def test_duplicate_ids_warning(schema):
    r = run(schema, make_csv(schema, override={("TransactionID", 1): "1000"}))
    assert not r.errors and any("повторяющиеся TransactionID" in w for w in r.warnings)


def test_without_id_column(schema):
    r = run(schema, make_csv(schema, with_id=False))
    assert not r.errors and r.ids is None
