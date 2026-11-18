"""Streamlit: прогноз мошенничества финальной моделью LightGBM II по ЗАРАНЕЕ ПОДГОТОВЛЕННЫМ 458 признакам.

Запуск локально:  set MODEL_PATH=<путь к lgbm_B_full.txt>  затем  streamlit run app.py
На сервере модель берётся из приватного репозитория Hugging Face Hub: HF_REPO_ID и HF_TOKEN — из переменных
окружения или секретов Streamlit (не из кода).
"""
import os

import streamlit as st

from fraud_app.config import ROOT, load_config, load_schema
from fraud_app.io_csv import read_csv_text
from fraud_app.model_loader import ModelError, load_booster, model_categories, resolve_model_path
from fraud_app.predict import predict_proba, result_table, results_file_name
from fraud_app.validation import prepare

st.set_page_config(page_title="LightGBM II: прогноз мошенничества", page_icon="🛡️", layout="wide")
CFG, SCHEMA = load_config(), load_schema()
DECISION = {1: "Тревога", 0: "Обычная операция"}


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


def screen_table(res):
    """Таблица для экрана: столбец «Решение» только для отображения (скачиваемый CSV не меняется)."""
    view = res.head(1000).copy()
    view.insert(view.columns.get_loc("predicted_class") + 1, "Решение", view["predicted_class"].map(DECISION))
    return view


# ---------------- заголовок и назначение ----------------
st.title("Прогноз мошенничества: LightGBM II")
st.markdown(
    "Приложение оценивает **вероятность мошенничества** для банковских операций с помощью обученной модели "
    "LightGBM II. Вы загружаете таблицу с подготовленными признаками операций и получаете для каждой операции "
    "вероятность мошенничества и решение: **обычная операция** или **тревога**."
)

# ---------------- как пользоваться ----------------
st.subheader("Как пользоваться")
s1, s2, s3 = st.columns(3)
with s1.container(border=True):
    st.markdown("**1. Подготовьте CSV**")
    st.markdown("Нужны 458 признаков с именами, как в шаблоне. Признаки строятся заранее тем же кодом, "
                "что при обучении модели.")
with s2.container(border=True):
    st.markdown("**2. Загрузите файл**")
    st.markdown("Приложение проверит столбцы и значения и сообщит об ошибках, если они есть.")
with s3.container(border=True):
    st.markdown("**3. Получите результаты**")
    st.markdown("Вероятность и класс для каждой операции, сводка по тревогам и файл для скачивания.")

# ---------------- важное ограничение ----------------
st.info(
    "**Исходные транзакции не принимаются.** Приложение не выполняет preprocessing: исходные CSV transaction "
    "и identity загрузить нельзя. Частотные признаки, история группы card1+addr1 и другие производные признаки "
    "должны быть построены заранее тем же кодом, что при обучении модели.",
    icon="ℹ️",
)

# ---------------- загрузка файла ----------------
st.subheader("Загрузка файла")
with st.container(border=True):
    st.markdown(
        f"""
- CSV в кодировке **UTF-8**, разделитель «,», десятичная точка «.»; не более {CFG['max_rows']:,} строк.
- Ровно **458 признаков** с именами из шаблона и, по желанию, `{SCHEMA.id_column}`. `isFraud` и другие лишние
  столбцы — ошибка.
- Пропуск — только **пустая ячейка**. Недостающие признаки не заполняются.
- Если порядок столбцов другой, приложение переставит их по именам.
"""
    )
    tmpl = ROOT / "templates" / "features_template.csv"
    st.download_button("Скачать шаблон заголовков (458 признаков)", tmpl.read_bytes(),
                       file_name="features_template.csv", mime="text/csv")
    up = st.file_uploader("CSV с подготовленными признаками", type=["csv"])
    with st.expander("Подробные правила формата"):
        st.markdown(
            """
- В числовых признаках допустимы только число или пустая ячейка; любой другой текст («NA», «nan», «inf», «1,5»)
  — ошибка с номерами строк.
- В категориальных признаках текст сохраняется как есть: например, «NA» — это значение, а не пропуск.
- Значение категории, которого не было при обучении, модель обрабатывает как пропуск (так же, как при оценке
  модели); приложение показывает, сколько таких значений.
- Числовые значения лучше записывать с полной точностью: при округлении прогноз может немного измениться.
"""
        )

results_area = st.container()   # результаты — под областью загрузки
about_area = st.container()     # «О модели» — внизу страницы

# ---------------- модель ----------------
try:
    booster, categories = get_model()
except ModelError as e:
    st.error(f"Модель не загружена: {e}")
    st.stop()

with about_area.expander("О модели"):
    st.caption(f"Модель проверена: SHA-256 {CFG['model_sha256'][:12]}…, деревьев {booster.num_trees()}, признаков 458.")
    st.markdown(
        f"""
- **Модель:** LightGBM II, {booster.num_trees()} деревьев, 458 признаков (41 категориальный, 417 числовых).
- **Контрольная сумма файла модели (SHA-256):** `{CFG['model_sha256']}` — проверяется при каждой загрузке.
- **Порог по умолчанию:** {CFG['default_threshold']!r}.
- **Данные:** соревнование IEEE-CIS Fraud Detection (Kaggle); обучение на первых 80% операций по времени.
- **Оценка на Validation (42 дня):** AP 0.621, ROC-AUC 0.919, F1 0.597 при пороге 0.0486. Это результаты на
  Validation, которая уже использовалась при разработке и сравнении моделей, — **не независимая оценка на Test**.
  Тестовая выборка (Test) для оценки качества модели не использовалась.
- **Ограничения:** один seed; порог выбран по F1 на внутренней проверке, стоимость ошибок неизвестна;
  переносимость на более поздние периоды не проверялась.
"""
    )

if up is None:
    st.stop()

# ---------------- результаты ----------------
with results_area:
    raw = read_csv_text(up.getvalue())
    prep = prepare(raw, SCHEMA, categories, CFG["max_rows"])
    for e in prep.errors:
        st.error(e)
    if prep.errors:
        st.stop()
    for w in prep.warnings:
        st.warning(w)

    st.subheader("Результаты")
    summary = st.container()   # метрики заполняются после выбора порога, но показываются первыми
    with st.expander("Как читать результаты"):
        st.markdown(
            """
- **Вероятность мошенничества** (`fraud_probability`) — оценка модели от 0 до 1. Чем выше, тем подозрительнее
  операция. Вероятности не откалиброваны: это оценка для сравнения с порогом, а не точная доля мошенничества.
- **Порог** — граница решения. По умолчанию 0.0486: порог выбран по F1 на внутренней проверке при обучении.
- **Класс 1 — «Тревога»:** вероятность ≥ порога, операцию стоит проверить.
  **Класс 0 — «Обычная операция»:** вероятность ниже порога.
- **Тревог** — число операций класса 1; **доля тревог** — их процент от всех операций.
- Порог можно изменить ниже. Качество модели при других порогах не проверялось.
"""
        )
    threshold = st.number_input("Порог классификации (по умолчанию 0.0486)", min_value=0.0, max_value=1.0,
                                value=float(CFG["default_threshold"]), step=0.001, format="%.4f")
    st.caption("Изменение порога меняет только классы; вероятности остаются прежними.")
    if abs(threshold - CFG["default_threshold"]) > 1e-12:
        st.info("Порог изменён. Качество модели при других порогах не проверялось.")

    p = predict_proba(booster, prep.X)
    res = result_table(p, threshold, prep.ids, SCHEMA.id_column)

    with summary:
        c1, c2, c3 = st.columns(3)
        c1.metric("Операций", f"{len(res):,}")
        c2.metric(f"Тревог (класс 1) при пороге {threshold:.4f}", f"{int(res['predicted_class'].sum()):,}")
        c3.metric("Доля тревог", f"{res['predicted_class'].mean():.2%}")
        st.caption(f"Фактический порог классификации: {threshold:.10g} (класс 1, если вероятность ≥ порога).")

    st.dataframe(
        screen_table(res), width="stretch", hide_index=True,
        column_config={
            "fraud_probability": st.column_config.NumberColumn(help="Вероятность мошенничества по модели, от 0 до 1"),
            "predicted_class": st.column_config.NumberColumn(help="1 — тревога (вероятность ≥ порога), 0 — нет"),
            "Решение": st.column_config.TextColumn(help="Только на экране; в скачиваемом CSV — столбец predicted_class"),
        },
    )
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
