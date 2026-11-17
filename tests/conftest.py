import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fraud_app.config import load_config, load_schema  # noqa: E402


@pytest.fixture(scope="session")
def schema():
    return load_schema()


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def booster(cfg, schema):
    """Настоящая модель — только если задан MODEL_PATH (локальная проверка). Иначе тесты с моделью пропускаются."""
    path = os.environ.get("MODEL_PATH")
    if not path:
        pytest.skip("MODEL_PATH не задан — тесты с моделью пропущены")
    from fraud_app.model_loader import load_booster
    return load_booster(Path(path), cfg, schema)


@pytest.fixture(scope="session")
def local_csv():
    """Локальные реальные признаки Validation — только на этом компьютере, вне репозитория."""
    d = os.environ.get("LOCAL_DATA_DIR")
    if not d or not (Path(d) / "validation_2000_features.csv").exists():
        pytest.skip("LOCAL_DATA_DIR не задан или нет локального файла — приёмочный тест пропущен")
    return Path(d)
