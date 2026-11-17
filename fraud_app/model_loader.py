"""Получение и проверка модели LightGBM II.

Модель берётся либо из локального файла (переменная окружения MODEL_PATH — для локальной проверки),
либо из приватного репозитория Hugging Face Hub по токену (только из секретов / окружения, не из кода).
"""
import hashlib
from pathlib import Path

import lightgbm as lgb

from .config import Schema


class ModelError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def resolve_repo_id(cfg: dict, repo_id: str | None = None) -> str | None:
    """Репозиторий модели: HF_REPO_ID (переменная окружения / секрет, передаётся сюда) важнее значения в конфиге.
    Заглушка в конфиге (начинается с «<») считается незаданным значением."""
    for v in (repo_id, cfg.get("hf_repo_id")):
        s = str(v).strip() if v is not None else ""
        if s and not s.startswith("<"):
            return s
    return None


def resolve_model_path(cfg: dict, local_path: str | None = None, token: str | None = None,
                       repo_id: str | None = None) -> Path:
    if local_path:
        p = Path(local_path)
        if not p.is_file():
            raise ModelError(f"файл модели не найден: {p}")
        return p
    if not token:
        raise ModelError("не задан ни MODEL_PATH, ни токен Hugging Face (HF_TOKEN)")
    rid = resolve_repo_id(cfg, repo_id)
    if not rid:
        raise ModelError("не задан репозиторий модели: переменная окружения или секрет HF_REPO_ID")
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id=rid, filename=cfg["model_filename"], repo_type="model", token=token))


def load_booster(path: Path, cfg: dict, schema: Schema) -> lgb.Booster:
    digest = sha256_file(path)
    if digest != cfg["model_sha256"]:
        raise ModelError(f"контрольная сумма модели не совпадает: {digest[:16]}… вместо {cfg['model_sha256'][:16]}…")
    booster = lgb.Booster(model_file=str(path))
    if booster.num_trees() != cfg["n_trees"]:
        raise ModelError(f"в модели {booster.num_trees()} деревьев вместо {cfg['n_trees']}")
    if tuple(booster.feature_name()) != schema.names:
        raise ModelError("имена или порядок признаков модели не совпадают со схемой")
    cats = booster.pandas_categorical or []
    if len(cats) != len(schema.categorical):
        raise ModelError(f"в модели {len(cats)} категориальных признаков вместо {len(schema.categorical)}")
    if not all(isinstance(v, str) for lst in cats for v in lst):
        raise ModelError("в словарях категорий модели есть нестроковые значения — нужна отдельная проверка чтения CSV")
    return booster


def model_categories(booster: lgb.Booster, schema: Schema) -> dict:
    """Словари категорий модели: признак -> список значений в порядке кодов LightGBM."""
    return dict(zip(schema.categorical, booster.pandas_categorical))
