"""Streamlit: прогноз мошенничества финальной моделью LightGBM II по ЗАРАНЕЕ ПОДГОТОВЛЕННЫМ 458 признакам.

Запуск локально:  set MODEL_PATH=<путь к lgbm_B_full.txt>  затем  streamlit run app.py
На сервере модель берётся из приватного репозитория Hugging Face Hub: HF_REPO_ID и HF_TOKEN — из переменных
окружения или секретов Streamlit (не из кода).
"""
import os
from pathlib import Path

import streamlit as st

from fraud_app.config import ROOT, load_config, load_schema
from fraud_app.io_csv import read_csv_text
from fraud_app.model_loader import ModelError, load_booster, model_categories, resolve_model_path
from fraud_app.predict import predict_proba, result_table, results_file_name
from fraud_app.validation import prepare

st.set_page_config(page_title="LightGBM II: прогноз мошенничества", layout="wide")
CFG, SCHEMA = load_config(), load_schema()


def _secret(name):
    try:
        return st.secrets[name]
    except Exception:  # noqa: BLE001 — секретов может не быть при локальном запуске
        return None


@st.cache_resource(show_spinner="Загрузка модели…")
def get_model():
    local = os.environ.get("MODEL_PATH")
    token = None if local else (os.environ.get("HF_TOKEN") or _secret("HF_TOKEN"))
    repo_id = None if local else (os.environ.get("HF_REPO_ID") or _secret("HF_REPO_ID"))
    path = resolve_model_path(CFG, local_path=local, token=token, repo_id=repo_id)
    booster = load_booster(path, CFG, SCHEMA)
    return booster, model_categories(booster, SCHEMA)


st.title("Прогноз мошенничества: LightGBM II")
st.warning(
    "**Приложение работает только с заранее подготовленными признаками (458 столбцов).** "
    "Полный preprocessing исходных транзакций здесь **не выполняется**: частотные признаки, история группы "
    "card1+addr1 и другие производные признаки должны быть построены тем же кодом, что при обучении модели. "
    "Исходные CSV транзакций и identity как входные данные не принимаются."
)

with st.expander("Формат файла и ограничения модели"):
    st.markdown(
        f"""
- CSV в кодировке **UTF-8**, разделитель «,», десятичная точка «.»; не более {CFG['max_rows']:,} строк.
- Ровно **458 признаков** с именами из шаблона; необязательный столбец `{SCHEMA.id_column}`. Другие столбцы
  (в том числе `isFraud`) — ошибка. Если порядок другой, столбцы переставляются по именам.
- **Отсутствующие признаки не заполняются** — это ошибка.
- Пропуск — только **пустая ячейка**. В числовых признаках любой другой нечисловой текст — ошибка.
  В категориальных текст сохраняется как есть (например, «NA» — это значение, а не пропуск).
- Значение категории, которого не было при обучении, модель обрабатывает как пропуск (так же, как при оценке
  модели); приложение показывает, сколько таких значений.
- Модель: LightGBM II, 4628 деревьев. Оценка на Validation (42 дня): AP 0.621, ROC-AUC 0.919, F1 0.597 при пороге 0.0486.
  Это результаты на Validation, которая уже использовалась при разработке и сравнении моделей, — **не независимая
  оценка на Test**. Тестовая выборка (Test) для оценки качества модели не использовалась.
- Порог 0.0486 выбран по F1 на внутренней проверке; другие пороги не проверялись. Стоимость ошибок неизвестна.
"""
    )

tmpl = ROOT / "templates" / "features_template.csv"
st.download_button("Скачать шаблон заголовков (458 признаков)", tmpl.read_bytes(), file_name="features_template.csv",
                   mime="text/csv")

try:
    booster, categories = get_model()
except ModelError as e:
    st.error(f"Модель не загружена: {e}")
    st.stop()
st.caption(f"Модель проверена: SHA-256 {CFG['model_sha256'][:12]}…, деревьев {booster.num_trees()}, признаков 458.")

threshold = st.number_input("Порог классификации (по умолчанию 0.0486)", min_value=0.0, max_value=1.0,
                            value=float(CFG["default_threshold"]), step=0.001, format="%.4f")
if abs(threshold - CFG["default_threshold"]) > 1e-12:
    st.info("Порог изменён. Качество модели при других порогах не проверялось.")

up = st.file_uploader("CSV с подготовленными признаками", type=["csv"])
if up is None:
    st.stop()

raw = read_csv_text(up.getvalue())
prep = prepare(raw, SCHEMA, categories, CFG["max_rows"])
for e in prep.errors:
    st.error(e)
if prep.errors:
    st.stop()
for w in prep.warnings:
    st.warning(w)

p = predict_proba(booster, prep.X)
res = result_table(p, threshold, prep.ids, SCHEMA.id_column)

c1, c2, c3 = st.columns(3)
c1.metric("Операций", f"{len(res):,}")
c2.metric(f"Тревог (класс 1) при пороге {threshold:.4f}", f"{int(res['predicted_class'].sum()):,}")
c3.metric("Доля тревог", f"{res['predicted_class'].mean():.2%}")
st.caption(f"Фактический порог классификации: {threshold:.10g} (класс 1, если вероятность ≥ порога).")

st.subheader("Результаты")
st.dataframe(res.head(1000), width="stretch", hide_index=True)
if len(res) > 1000:
    st.caption("Показаны первые 1 000 строк; полный результат — в файле.")
st.download_button("Скачать результаты (CSV)", res.to_csv(index=False).encode("utf-8"),
                   file_name=results_file_name(CFG["model_sha256"], threshold), mime="text/csv")

with st.expander("Проверка входных данных"):
    if prep.unknown_categories is not None and len(prep.unknown_categories):
        st.markdown("**Незнакомые значения категорий** (модель обрабатывает их как пропуск):")
        st.dataframe(prep.unknown_categories, hide_index=True)
    else:
        st.markdown("Незнакомых значений категорий нет.")
    st.markdown("**Доля пропусков по признакам** (только признаки с пропусками, первые 30):")
    st.dataframe(prep.missing_share.head(30), hide_index=True)
