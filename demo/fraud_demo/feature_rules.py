"""Правила производных признаков — ДОСЛОВНО из исследовательского проекта (не новые правила).

Источники:
- ноутбук 01_2 (01_2_feature_engineering.ipynb): OS_RULES, BRAND_RULES, model_family_raw, transform_device,
  PROVIDER_RULES, transform_email, CARD_TYPE / CARD_NETWORK, transform_card, AMOUNT_BINS / AMOUNT_LABELS, transform_amount;
- ноутбук 13 (13_controlled_comparison_B.ipynb, ячейка «1. загрузка данных»): hour, weekday, n_missing_V/D/M/id
  (столбцы с префиксом и числовым хвостом), group_first_operation, интервалы истории (pd.cut, right=False).

Здесь только преобразования, нужные для синтетического набора демо. Параметры (частоты доменов и т. п.)
в коде не хранятся — они читаются из локальных приватных файлов проекта (см. synthetic.load_params).
"""
import re

import numpy as np
import pandas as pd

# ---------------- 01_2: устройство ----------------
OS_RULES = [
    ("Windows", r"^Windows", ""), ("macOS", r"^MacOS", ""), ("iOS", r"^(iOS Device|iPhone|iPad)", ""),
    ("Android", r"Build/|android", ""), ("Linux", r"^Linux", ""),
]
BRAND_RULES = [
    ("Apple", r"^(iOS Device|iPhone|iPad|MacOS)", ""), ("Samsung", r"samsung|^SM-", ""), ("Huawei", r"huawei", ""),
    ("LG", r"^LG", ""), ("Motorola", r"^moto|motorola", ""), ("Xiaomi", r"^redmi|^mi |xiaomi", ""),
    ("Google", r"^pixel", ""), ("HTC", r"^htc", ""), ("Lenovo", r"^lenovo", ""), ("ZTE", r"^zte", ""),
    ("Sony", r"^sony", ""), ("Alcatel", r"^alcatel", ""), ("ASUS", r"^asus", ""), ("Nokia", r"^nokia", ""),
    ("Hisense", r"^hisense", ""),
]


def match_rules(x, rules):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "Missing"
    for label, pattern, _ in rules:
        if re.search(pattern, x, flags=re.IGNORECASE):
            return label
    return "Unknown"


def model_family_raw(x, brand):
    if brand in ("Missing",):
        return "Missing"
    rules = [
        ("Samsung", r"^(?:SAMSUNG[- ])?SM-([A-Z])", "Samsung SM-{}"),
        ("Motorola", r"^Moto ([A-Z])", "Motorola Moto {}"),
        ("LG", r"^LG-?([A-Z])", "LG {}"),
        ("Huawei", r"^(?:HUAWEI )?([A-Z]{3})-[A-Z]", "Huawei {}"),
        ("Xiaomi", r"^(Redmi|Mi) ", "Xiaomi {}"),
        ("Google", r"^(Pixel)", "Google {}"),
    ]
    for b, p, fmt in rules:
        if brand == b:
            m = re.search(p, x, flags=re.IGNORECASE)
            if m:
                return fmt.format(m.group(1))
    return "Unknown"


def device_raw(df):
    s = df["DeviceInfo"].astype(object)
    uniq = s.dropna().unique()
    os_map = {v: match_rules(v, OS_RULES) for v in uniq}
    brand_map = {v: match_rules(v, BRAND_RULES) for v in uniq}
    fam_map = {v: model_family_raw(v, brand_map[v]) for v in uniq}
    # .astype(object) перед fillna: если DeviceInfo пуст во всех строках, map() даёт float64 и pandas предупреждает
    # о строковом fillna; результат правила 01_2 от этого не меняется
    return pd.DataFrame({
        "device_os": s.map(os_map).astype(object).fillna("Missing"),
        "device_brand": s.map(brand_map).astype(object).fillna("Missing"),
        "device_model_family": s.map(fam_map).astype(object).fillna("Missing"),
    }, index=df.index)


def transform_device(df, p):
    raw = device_raw(df)
    out = pd.DataFrame(index=df.index)
    out["device_os"] = raw["device_os"]
    special = {"Missing", "Unknown"}
    out["device_brand"] = raw["device_brand"].where(raw["device_brand"].isin(set(p["brands"]) | special), "Other")
    out["device_model_family"] = raw["device_model_family"].where(
        raw["device_model_family"].isin(set(p["families"]) | special), "Other")
    return out


# ---------------- 01_2: почта ----------------
PROVIDER_RULES = [
    ("Google", r"^gmail(\.com)?$"), ("Yahoo", r"^(yahoo\.|ymail\.com$|rocketmail\.com$)"),
    ("Microsoft", r"^(hotmail\.|outlook\.|live\.|msn\.com$)"), ("Apple", r"^(icloud\.com|me\.com|mac\.com)$"),
    ("AOL", r"^(aol\.com|aim\.com)$"), ("anonymous", r"^anonymous\.com$"),
]


def provider_rule(d):
    for name, p in PROVIDER_RULES:
        if re.search(p, d):
            return name
    return None


def transform_email(df, p):
    out = pd.DataFrame(index=df.index)
    for c, pref in [("P_emaildomain", "P"), ("R_emaildomain", "R")]:
        s = df[c].astype(object)
        uniq = s.dropna().unique()
        frequent = set(p["frequent_domains"][c])
        mapping = {d: (provider_rule(d) or (d if d in frequent else "Other")) for d in uniq}
        out[f"{pref}_email_provider"] = s.map(mapping).fillna("Missing")
        out[f"{pref}_email_domain_freq"] = s.fillna("Missing").map(p["domain_freq"][c]).fillna(0.0).astype("float32")
    both = df["P_emaildomain"].notna() & df["R_emaildomain"].notna()
    same = df["P_emaildomain"].astype(object) == df["R_emaildomain"].astype(object)
    out["email_same_domain"] = pd.Series(np.where(both, same.astype(float), np.nan), index=df.index).astype("float32")
    return out


# ---------------- 01_2: карта и сумма ----------------
CARD_TYPE = {"debit": "debit", "credit": "credit", "debit or credit": "other", "charge card": "other"}
CARD_NETWORK = ["visa", "mastercard", "american express", "discover"]


def transform_card(df):
    out = pd.DataFrame(index=df.index)
    c6 = df["card6"].astype(object)
    out["card_type_group"] = c6.map(CARD_TYPE).where(c6.notna(), "Missing").fillna("other")
    c4 = df["card4"].astype(object)
    out["card_category_group"] = c4.where(c4.isin(CARD_NETWORK) | c4.isna(), "other").fillna("Missing")
    return out


AMOUNT_BINS = [0, 10, 25, 50, 100, 200, 500, 1000, np.inf]
AMOUNT_LABELS = ["<10", "10–25", "25–50", "50–100", "100–200", "200–500", "500–1000", "≥1000"]


def transform_amount(df, p):
    a = df["TransactionAmt"].astype("float64")
    out = pd.DataFrame(index=df.index)
    out["amount_group"] = pd.cut(a, bins=AMOUNT_BINS, labels=AMOUNT_LABELS, right=False).astype(str)
    out["is_large_amount"] = (a > p["large_amount_threshold"]).astype("int8")
    return out


# ---------------- ноутбук 13 ----------------
def digit_cols(names, prefix):
    """Столбцы с префиксом и числовым хвостом: V1–V339, D1–D15, M1–M9, id_01–id_38 (DeviceType/DeviceInfo — нет)."""
    return [c for c in names if c.startswith(prefix) and c[len(prefix):].isdigit()]


def row_features(df, raw_cols):
    out = pd.DataFrame(index=df.index)
    out["hour"] = ((df["TransactionDT"] // 3600) % 24).astype("int8")
    out["weekday"] = ((df["TransactionDT"] // 86_400) % 7).astype("int8")
    for p in ("V", "D", "M", "id_"):
        out[f"n_missing_{p.rstrip('_')}"] = df[digit_cols(raw_cols, p)].isna().sum(axis=1)
    return out


def history_derived(df, spec):
    H = spec["history_from_01_3"]
    edges = lambda v: [np.inf if e == "inf" else float(e) for e in v]  # noqa: E731
    cnt = df["group_prev_transactions"]
    has_hist = cnt.fillna(0) > 0
    out = pd.DataFrame(index=df.index)
    out["group_first_operation"] = (cnt == 0).astype("int8")
    out["group_time_since_prev_bin"] = (pd.cut(df["group_time_since_prev"], edges(H["group_time_since_prev_bin"]["edges_sec"]),
                                               labels=H["group_time_since_prev_bin"]["labels"], right=False)
                                        .astype(object).where(has_hist, np.nan))
    out["group_prev_amount_mean_bin"] = (pd.cut(df["group_prev_amount_mean"], edges(H["group_prev_amount_mean_bin"]["edges_usd"]),
                                                labels=H["group_prev_amount_mean_bin"]["labels"], right=False)
                                         .astype(object).where(has_hist, np.nan))
    return out
