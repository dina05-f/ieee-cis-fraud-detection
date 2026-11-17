"""Прогноз LightGBM II и таблица результатов."""
import numpy as np
import pandas as pd


def predict_proba(booster, X: pd.DataFrame) -> np.ndarray:
    # тот же вызов, что в ноутбуках 13–14: predict(DataFrame с category-столбцами)
    return booster.predict(X)


def results_file_name(model_sha256: str, threshold: float) -> str:
    """Имя файла результатов: версия модели (8 символов SHA-256) и фактический порог (4 знака)."""
    return f"lgbm2_results_{model_sha256[:8]}_thr{threshold:.4f}.csv"


def result_table(p: np.ndarray, threshold: float, ids: pd.Series | None, id_column: str) -> pd.DataFrame:
    out = pd.DataFrame({"row": np.arange(1, len(p) + 1)})
    if ids is not None:
        out[id_column] = ids.to_numpy()
    out["fraud_probability"] = p
    out["predicted_class"] = (p >= threshold).astype(int)
    out["threshold"] = threshold
    return out
