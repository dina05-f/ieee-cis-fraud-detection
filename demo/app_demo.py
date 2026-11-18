"""Anti-Fraud — демонстрация: синтетические операции, настоящая модель LightGBM II. Запускается сразу, без кнопок.

Запуск (из корня репозитория):
    $env:MODEL_PATH = "<путь к lgbm_B_full.txt>"
    $env:FRAUD_PARAMS_DIR = "<папка с interpretable_features_params.json и lr_feature_spec.json>"
    .venv\\Scripts\\streamlit run demo\\app_demo.py --server.address 127.0.0.1 --server.port 8502

Только локально: модель берётся из MODEL_PATH, облако не используется.
Основное приложение (app.py) и пакет fraud_app не изменяются — демо только импортирует их функции.
"""
import os
import sys
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for p in (REPO, Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import streamlit as st  # noqa: E402

from fraud_app.config import load_config, load_schema  # noqa: E402
from fraud_app.model_loader import ModelError  # noqa: E402
from fraud_demo import charts, pipeline, synthetic  # noqa: E402

st.set_page_config(page_title="Anti-Fraud — демонстрация", page_icon="🛡️", layout="wide")
st.markdown("<style>.block-container{padding-top:2rem;padding-bottom:2rem}</style>", unsafe_allow_html=True)
CFG, SCHEMA = load_config(), load_schema()


@st.cache_resource(show_spinner="Загрузка модели LightGBM II…")
def get_model():
    """Модель загружается один раз на процесс (проверка SHA-256, числа деревьев и признаков — в fraud_app)."""
    return pipeline.load_model(os.environ.get("MODEL_PATH"), CFG, SCHEMA)


@st.cache_resource(show_spinner=False)
def get_params():
    return synthetic.load_params(os.environ.get("FRAUD_PARAMS_DIR"))


# ---------------- шапка ----------------
st.title("🛡️ Anti-Fraud")
st.badge("Демонстрация на синтетических данных", icon=":material/science:", color="orange")
st.markdown(
    "Система оценивает **риск мошенничества** для каждой банковской операции и отмечает подозрительные операции "
    "для ручной проверки. Ниже — проверка синтетических операций: данные созданы генератором, а риск рассчитала "
    "**настоящая обученная модель LightGBM II**."
)

# ---------------- прогон демо (один раз на сессию: обновление страницы = новая сессия = новый прогон) ----------------
try:
    booster = get_model()
    params, spec = get_params()
    if "demo" not in st.session_state:
        with st.spinner("Модель проверяет синтетические операции…"):
            st.session_state["demo"] = pipeline.run_demo(booster, CFG, SCHEMA, params, spec)
            st.session_state["demo_run_id"] = uuid.uuid4().hex
except (ModelError, synthetic.DemoDataError) as e:
    st.error(f"Демонстрацию не удалось запустить: {e}")
    st.stop()

demo = st.session_state["demo"]
thr_pct = f"{demo.threshold * 100:.2f}".replace(".", ",")

# ---------------- сводка ----------------
m1, m2, m3 = st.columns(3)
m1.metric("Проверено операций", f"{demo.n_total:,}".replace(",", " "), border=True)
m2.metric("🔴 Подозрительные операции", f"{demo.n_suspicious:,}".replace(",", " "), border=True,
          help=f"Вероятность мошенничества не ниже порога {thr_pct}% — такие операции стоит проверить вручную.")
m3.metric("🟢 Низкий риск", f"{demo.n_low:,}".replace(",", " "), border=True,
          help=f"Вероятность мошенничества ниже порога {thr_pct}%.")

# ---------------- график и таблица ----------------
left, right = st.columns([5, 7], gap="large")
with left:
    st.subheader("Распределение риска")
    p = demo.results["fraud_probability"].to_numpy()
    st.altair_chart(charts.risk_chart(pipeline.risk_bands(p, demo.threshold), demo.threshold), width="stretch")
    st.caption(f"Сколько операций попало в каждый диапазон вероятности. Красные столбцы — выше порога {thr_pct}%: "
               "эти операции система отмечает как подозрительные.")
with right:
    st.subheader("Результаты проверки")
    st.dataframe(
        pipeline.display_table(demo.results, SCHEMA.id_column), hide_index=True, width="stretch", height=380,
        column_config={
            "Номер операции": st.column_config.TextColumn(help="Синтетический номер операции"),
            "Уровень риска": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format=" ",
                                                             help="Наглядная шкала вероятности мошенничества от 0 до 100%"),
            "Вероятность мошенничества": st.column_config.NumberColumn(format="%.2f%%",
                                                                       help="Оценка модели LightGBM II, от 0 до 100%"),
            "Результат проверки": st.column_config.TextColumn(
                help=f"Подозрительная — вероятность не ниже порога {thr_pct}%; низкий риск — ниже порога"),
        },
    )
    st.caption("Операции отсортированы от самого высокого риска к самому низкому.")

# ---------------- пояснения ----------------
st.markdown(
    f"""
**Как это работает.** Для каждой операции модель LightGBM II оценивает вероятность мошенничества. Если вероятность
не ниже порога **{thr_pct}%**, операция отмечается как **подозрительная** — её стоит проверить вручную. Остальные
операции — **низкий риск**.
"""
)
st.info(
    "**Это демонстрация.** Операции синтетические: они созданы генератором и не являются реальными банковскими "
    "транзакциями или данными клиентов. Поэтому числа на этой странице показывают, как работает система, "
    "а не долю мошенничества в реальных платежах.",
    icon="ℹ️",
)
with st.expander("О модели и данных"):
    st.markdown(
        f"""
- **Модель:** LightGBM II — {booster.num_trees()} деревьев, 458 признаков. Файл модели проверен по контрольной сумме
  (SHA-256 `{CFG['model_sha256'][:12]}…`). Модель не переобучалась и не изменялась.
- **Порог:** {demo.threshold!r} (≈ {thr_pct}%) — тот же, что в основном приложении.
- **Синтетический набор:** {demo.n_total} операций, фиксированное начальное значение генератора ({synthetic.SEED}),
  поэтому результат воспроизводим. Значения категорий взяты только из словарей самой модели; редкие значения
  (модель устройства, браузер, версия ОС, разрешение экрана) не используются — эти поля оставлены пустыми.
  Производные признаки рассчитаны по тем же правилам, что при обучении.
- **Вероятность** — оценка модели для сравнения с порогом; она не откалибрована и не равна точной доле мошенничества.
- Качество модели оценивалось в исследовательском проекте на отложенной выборке реальных данных соревнования;
  на синтетических данных качество не измеряется.
"""
    )
