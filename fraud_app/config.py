"""Чтение настроек приложения и схемы признаков."""
import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Schema:
    names: tuple            # 458 признаков в порядке обучения
    categorical: tuple      # 41 категориальный признак (в том же порядке, что в names)
    id_column: str

    @property
    def numeric(self):
        cat = set(self.categorical)
        return tuple(n for n in self.names if n not in cat)


def load_config() -> dict:
    return json.loads((ROOT / "config" / "app_config.json").read_text(encoding="utf-8"))


def load_schema() -> Schema:
    s = json.loads((ROOT / "config" / "feature_schema.json").read_text(encoding="utf-8"))
    names = tuple(f["name"] for f in s["features"])
    cats = tuple(f["name"] for f in s["features"] if f["kind"] == "categorical")
    assert len(names) == s["n_features"] == 458 and len(set(names)) == 458
    assert len(cats) == s["n_categorical"] == 41
    return Schema(names=names, categorical=cats, id_column=s["id_column"])
