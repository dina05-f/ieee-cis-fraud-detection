"""Точка входа демо для Streamlit Community Cloud.

Модель LightGBM II и параметры признаков хранятся в приватном репозитории Hugging Face Hub. При старте они
скачиваются по секретам HF_REPO_ID и HF_TOKEN (токен только на чтение), затем запускается demo/app_demo.py.
Целостность модели (SHA-256, число деревьев, имена признаков) проверяет fraud_app, как и в основном приложении.
Локально можно не использовать: задайте MODEL_PATH и FRAUD_PARAMS_DIR и запускайте demo/app_demo.py напрямую.
"""
import os
import runpy
from pathlib import Path

import streamlit as st

PARAMS = ("params/interpretable_features_params.json", "params/lr_feature_spec.json")


def _secret(name):
    value = os.environ.get(name)
    if value:
        return value
    try:
        return st.secrets.get(name)
    except Exception:  # секретов нет (локальный запуск без secrets.toml)
        return None


if not (os.environ.get("MODEL_PATH") and os.environ.get("FRAUD_PARAMS_DIR")):
    try:
        from huggingface_hub import hf_hub_download

        repo_id, token = _secret("HF_REPO_ID"), _secret("HF_TOKEN")
        if not repo_id:
            raise RuntimeError("не задан секрет HF_REPO_ID")
        model = hf_hub_download(repo_id, "lgbm_B_full.txt", token=token)
        params = [hf_hub_download(repo_id, p, token=token) for p in PARAMS]
        os.environ["MODEL_PATH"] = model
        os.environ["FRAUD_PARAMS_DIR"] = str(Path(params[0]).parent)
    except Exception as e:  # без доступа к модели — понятное сообщение, без трассировки и без значений секретов
        st.set_page_config(page_title="Anti-Fraud — демонстрация", page_icon="🛡️")
        st.error("Не удалось загрузить модель из приватного репозитория Hugging Face. Проверьте секреты HF_REPO_ID "
                 f"и HF_TOKEN в настройках приложения. Тип ошибки: {type(e).__name__}")
        st.stop()

runpy.run_path(str(Path(__file__).resolve().parent / "app_demo.py"), run_name="__main__")
