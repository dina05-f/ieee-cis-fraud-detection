"""Тесты демо. Нужны локальная модель (MODEL_PATH) и параметры признаков (FRAUD_PARAMS_DIR); без них тесты пропускаются."""
import os
import sys
from pathlib import Path

import pytest

DEMO = Path(__file__).resolve().parent.parent
REPO = DEMO.parent
for p in (REPO, DEMO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fraud_app.config import load_config, load_schema  # noqa: E402


@pytest.fixture(scope="session")
def schema():
    return load_schema()


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def booster(cfg, schema):
    if not os.environ.get("MODEL_PATH"):
        pytest.skip("MODEL_PATH не задан — тесты демо с моделью пропущены")
    from fraud_demo.pipeline import load_model
    return load_model(os.environ["MODEL_PATH"], cfg, schema)


@pytest.fixture(scope="session")
def params():
    if not os.environ.get("FRAUD_PARAMS_DIR"):
        pytest.skip("FRAUD_PARAMS_DIR не задан — тесты демо пропущены")
    from fraud_demo.synthetic import load_params
    return load_params(os.environ["FRAUD_PARAMS_DIR"])


@pytest.fixture(scope="session")
def demo(booster, cfg, schema, params):
    from fraud_demo.pipeline import run_demo
    return run_demo(booster, cfg, schema, *params)
