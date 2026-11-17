"""Проверка таблицы признаков и подготовка входа модели — так же, как в ноутбуках 13–14.

Как обучалась и оценивалась LightGBM II (ноутбуки 13–14):
- категориальные признаки передавались в LightGBM как pandas-столбцы типа category со строковыми значениями;
- LightGBM при прогнозе сам приводит категории к словарям обучения (booster.pandas_categorical, set_categories),
  поэтому значение, которого не было в обучении, становится пропуском (код -1 -> NaN);
- пропуски передаются как NaN и обрабатываются правилом ветвления модели.
Приложение повторяет именно этот путь: строит DataFrame с category-столбцами и вызывает booster.predict(DataFrame).
Значения не подменяются; незнакомые категории только подсчитываются для отчёта.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Schema
from .io_csv import RawTable, parse_categorical, parse_numeric


@dataclass
class Prepared:
    X: pd.DataFrame | None
    ids: pd.Series | None
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    unknown_categories: pd.DataFrame | None = None
    missing_share: pd.DataFrame | None = None


def prepare(raw: RawTable, schema: Schema, categories: dict, max_rows: int) -> Prepared:
    res = Prepared(None, None, list(raw.errors))
    if res.errors:
        return res
    header, names, idc = raw.header, schema.names, schema.id_column
    missing = [n for n in names if n not in header]
    extra = [c for c in header if c not in set(names) and c != idc]
    if missing:
        res.errors.append(f"нет {len(missing)} из 458 признаков (нулями не заполняются): {missing[:15]}"
                          + (" …" if len(missing) > 15 else ""))
    if extra:
        hint = " Столбец isFraud (истинная метка) в приложение не подаётся." if "isFraud" in extra else ""
        res.errors.append(f"лишние столбцы (разрешён только {idc}): {extra[:15]}" + (" …" if len(extra) > 15 else "") + hint)
    n = len(raw.text)
    if n == 0:
        res.errors.append("в файле нет строк с данными")
    if n > max_rows:
        res.errors.append(f"строк {n:,} — больше лимита {max_rows:,}")
    if res.errors:
        return res

    feat_in_file = [c for c in header if c != idc]
    if feat_in_file != list(names):
        res.warnings.append("порядок признаков в файле отличается от порядка обучения — столбцы переставлены по именам")

    cols, bad_report = {}, []
    for name in names:
        s = raw.text[name]
        if name in schema.categorical:
            cols[name] = parse_categorical(s)
        else:
            v, bad = parse_numeric(s)
            if bad.any():
                rows = (np.flatnonzero(bad.to_numpy())[:5] + 2).tolist()          # номер строки файла (заголовок — 1)
                bad_report.append(f"{name}: строки {rows}, пример «{s[bad].iloc[0]}»")
            cols[name] = v
    if bad_report:
        res.errors.append(f"нечисловые значения в {len(bad_report)} числовых признаках (автозамены нет): "
                          + "; ".join(bad_report[:10]))
        return res

    X = pd.DataFrame(cols, index=raw.text.index)
    for c in schema.categorical:
        X[c] = X[c].astype("category")
    X = X[list(names)]
    # после перестановки — точное совпадение с моделью
    assert tuple(X.columns) == names
    assert tuple(c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)) == schema.categorical

    rows_unknown = []
    for c in schema.categorical:
        s = X[c]
        known = pd.Index(categories[c])
        unk = s.notna() & ~s.astype(object).isin(known)
        if unk.any():
            ex = pd.Series(s[unk].astype(object)).value_counts().head(3)
            rows_unknown.append({"признак": c, "строк с незнакомым значением": int(unk.sum()),
                                 "примеры": ", ".join(f"«{k}» ({v})" for k, v in ex.items())})
    res.unknown_categories = pd.DataFrame(rows_unknown, columns=["признак", "строк с незнакомым значением", "примеры"])
    miss = X.isna().mean()
    res.missing_share = (miss[miss > 0].sort_values(ascending=False).rename("доля пропусков").reset_index()
                         .rename(columns={"index": "признак"}))
    if idc in header:
        ids = raw.text[idc]
        if ids.duplicated().any():
            res.warnings.append(f"повторяющиеся {idc}: {int(ids.duplicated().sum())} строк")
        res.ids = ids
    res.X = X
    return res
