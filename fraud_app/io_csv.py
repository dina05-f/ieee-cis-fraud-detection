"""Чтение CSV с подготовленными признаками.

Правила:
- кодировка UTF-8 (допускается BOM); разделитель «,», десятичная точка «.»;
- все ячейки сначала читаются как текст, автоматического распознавания пропусков нет (keep_default_na=False):
  тексты «NA», «None», «null», «nan», «–», «≥…» остаются текстом;
- пропуском считается только пустая ячейка;
- числовой признак: пустая ячейка -> NaN; иначе значение должно быть конечным числом, любой другой текст
  (в том числе «NA», «nan», «inf») — ошибка, без автозамены;
- категориальный признак: пустая ячейка -> NaN; любой непустой текст сохраняется как есть (без обрезки пробелов).
"""
import csv
import io
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class RawTable:
    header: list
    text: pd.DataFrame                     # все значения — строки, пустая ячейка = ""
    errors: list = field(default_factory=list)


def read_csv_text(data: bytes) -> RawTable:
    try:
        content = data.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        return RawTable([], pd.DataFrame(), [f"файл не в кодировке UTF-8 ({e.reason}, байт {e.start})"])
    if not content.strip():
        return RawTable([], pd.DataFrame(), ["файл пустой"])
    header = next(csv.reader(io.StringIO(content)))
    dup = sorted({c for c in header if header.count(c) > 1})
    if dup:
        return RawTable(header, pd.DataFrame(), [f"повторяющиеся имена столбцов: {dup[:10]}"])
    try:
        text = pd.read_csv(io.StringIO(content), dtype=str, keep_default_na=False, na_values=[], sep=",",
                           encoding=None, engine="c")
    except Exception as e:  # noqa: BLE001 — сообщение пользователю
        return RawTable(header, pd.DataFrame(), [f"не удалось прочитать CSV: {e}"])
    if list(text.columns) != header:
        return RawTable(header, pd.DataFrame(), ["не удалось однозначно прочитать заголовок CSV"])
    return RawTable(header, text)


def parse_numeric(s: pd.Series):
    """Текст -> float64. Возвращает (значения, маска ошибочных строк)."""
    empty = s == ""
    num = pd.to_numeric(s.where(~empty), errors="coerce").astype("float64")
    bad = ~empty & (num.isna() | ~np.isfinite(num.fillna(0.0)))
    return num, bad


def parse_categorical(s: pd.Series) -> pd.Series:
    """Текст -> object: пустая ячейка -> NaN, остальное как есть."""
    out = s.astype(object)
    return out.where(s != "", np.nan)
