"""Выбор источника модели: MODEL_PATH, HF_REPO_ID, заглушка в конфиге. Без сети: загрузка с Hugging Face подменяется."""
from pathlib import Path

import pytest

from fraud_app import model_loader
from fraud_app.model_loader import ModelError, resolve_model_path, resolve_repo_id

PLACEHOLDER = {"hf_repo_id": "<заглушка>", "model_filename": "lgbm_B_full.txt"}


def test_placeholder_is_not_a_repo():
    assert resolve_repo_id(PLACEHOLDER) is None
    assert resolve_repo_id(PLACEHOLDER, "  ") is None


def test_env_value_overrides_config():
    assert resolve_repo_id(PLACEHOLDER, "owner/repo") == "owner/repo"
    assert resolve_repo_id({"hf_repo_id": "cfg/repo"}, "env/repo") == "env/repo"
    assert resolve_repo_id({"hf_repo_id": "cfg/repo"}, None) == "cfg/repo"


def test_errors_without_token_or_repo():
    with pytest.raises(ModelError, match="HF_TOKEN"):
        resolve_model_path(PLACEHOLDER, token=None, repo_id="owner/repo")
    with pytest.raises(ModelError, match="HF_REPO_ID"):
        resolve_model_path(PLACEHOLDER, token="t", repo_id=None)


def test_local_path_has_priority(tmp_path):
    f = tmp_path / "m.txt"
    f.write_text("x")
    assert resolve_model_path(PLACEHOLDER, local_path=str(f), token=None) == f
    with pytest.raises(ModelError, match="не найден"):
        resolve_model_path(PLACEHOLDER, local_path=str(tmp_path / "нет.txt"))


def test_hf_download_called_with_env_repo(monkeypatch, tmp_path):
    calls = {}

    def fake_download(**kw):
        calls.update(kw)
        return str(tmp_path / "downloaded.txt")

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    p = resolve_model_path(PLACEHOLDER, token="secret-token", repo_id="owner/private-repo")
    assert p == Path(tmp_path / "downloaded.txt")
    assert calls == {"repo_id": "owner/private-repo", "filename": "lgbm_B_full.txt", "repo_type": "model",
                     "token": "secret-token"}
    assert model_loader  # модуль импортируется без сети
