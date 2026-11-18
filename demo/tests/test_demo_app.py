"""Интерфейс демо (streamlit AppTest, без браузера): запуск без кнопок, результаты сразу, без повторной генерации."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

from fraud_demo import pipeline

APP = str(Path(__file__).resolve().parent.parent / "app_demo.py")


def _open(booster, params):  # фикстуры — чтобы тест пропускался без MODEL_PATH / FRAUD_PARAMS_DIR
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception, at.exception
    return at


def test_opens_with_results_and_no_start_button(booster, params, demo):
    at = _open(booster, params)
    assert not at.error
    assert len(at.button) == 0 and len(at.get("download_button")) == 0 and len(at.file_uploader) == 0
    assert any("Anti-Fraud" in t.value for t in at.title)
    assert any("Демонстрация на синтетических данных" in md.value for md in at.markdown)   # st.badge = markdown
    m = {x.label: x.value for x in at.metric}
    assert m["Проверено операций"] == str(demo.n_total)
    assert m["🔴 Подозрительные операции"] == str(demo.n_suspicious)
    assert m["🟢 Низкий риск"] == str(demo.n_low)
    assert len(at.get("vega_lite_chart")) == 1                                   # график распределения риска
    t = at.dataframe[0].value
    assert list(t.columns) == ["Номер операции", "Результат проверки", "Вероятность мошенничества", "Уровень риска"]
    assert t.equals(pipeline.display_table(demo.results, "TransactionID"))      # те же прогнозы, тот же порядок


def test_rerun_does_not_regenerate(booster, params):
    at = _open(booster, params)
    run_id = at.session_state["demo_run_id"]
    at.run()
    assert not at.exception and at.session_state["demo_run_id"] == run_id


def test_new_session_runs_demo_again_with_same_result(booster, params):
    a, b = _open(booster, params), _open(booster, params)
    assert a.session_state["demo_run_id"] != b.session_state["demo_run_id"]      # новый прогон
    assert a.dataframe[0].value.equals(b.dataframe[0].value)                     # фиксированный seed -> тот же итог
