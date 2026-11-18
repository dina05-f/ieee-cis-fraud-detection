"""График распределения риска (Altair, входит в Streamlit).

Форма: столбцы — число операций в диапазонах вероятности (ось X — упорядоченные диапазоны, одна ось Y).
Цвет несёт статус и всегда продублирован подписью: легенда «Низкий риск / Подозрительная», подписи диапазонов на оси,
а в таблице — текстовый результат и значок. Красный — статусный «critical» из палитры, для низкого риска —
нейтральный серый (не статусный зелёный: так пара различима и при нарушениях цветового зрения по светлоте).
"""
import altair as alt

from .pipeline import RESULT_LOW, RESULT_SUSPICIOUS

COLOR_LOW = "#a3a29b"         # нейтральный серый
COLOR_SUSPICIOUS = "#d03b3b"  # статус critical
INK_MUTED, GRID = "#898781", "#e1e0d9"


def risk_chart(bands, threshold):
    order = bands.sort_values("Порядок")["Диапазон вероятности"].tolist()   # от низкой вероятности к высокой
    base = alt.Chart(bands)
    bars = base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("Диапазон вероятности:N", sort=order, title=None,
                axis=alt.Axis(labelAngle=0, labelColor=INK_MUTED, domainColor=GRID, tickColor=GRID, labelFontSize=11)),
        y=alt.Y("Операций:Q", title="Операций", axis=alt.Axis(labelColor=INK_MUTED, titleColor=INK_MUTED,
                                                                gridColor=GRID, domain=False, ticks=False, tickMinStep=1)),
        color=alt.Color("Результат:N", scale=alt.Scale(domain=[RESULT_LOW, RESULT_SUSPICIOUS],
                                                        range=[COLOR_LOW, COLOR_SUSPICIOUS]),
                        legend=alt.Legend(title=None, orient="top", labelColor="#52514e", symbolType="square")),
        tooltip=[alt.Tooltip("Диапазон вероятности:N", title="Вероятность"),
                 alt.Tooltip("Операций:Q", title="Операций"), alt.Tooltip("Результат:N", title="Результат")],
    )
    labels = base.mark_text(dy=-7, fontSize=11, color="#52514e").encode(
        x=alt.X("Диапазон вероятности:N", sort=order), y="Операций:Q", text="Операций:Q")
    return (bars + labels).properties(height=300).configure_view(stroke=None).configure_scale(bandPaddingInner=0.15)
