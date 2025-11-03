import pandas as pd
import plotly.express as px


def _score_from_series(series: pd.Series, labels, ascending: bool = True) -> pd.Series:
    """
    Devuelve un puntaje por cuantiles usando qcut sobre el RANK.
    Al usar rank, todos los valores son únicos y evitamos el problema
    de 'Bin labels must be one fewer than the number of bin edges'.
    """
    s = series.copy()
    # Rango 1..n (todos únicos)
    rank = s.rank(method="first", ascending=ascending)
    # Siempre usamos tantos labels como cuantiles
    return pd.qcut(rank, q=len(labels), labels=labels)


def calcular_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula Recency, Frequency y Monetary por usuario,
    más el score RFM y el segmento.
    Usa la columna 'ingresos_cop' como valor monetario.
    """
    df = df.copy()

    for col in ["user_id", "start_date_time", "ingresos_cop"]:
        if col not in df.columns:
            raise ValueError(f"El dataset no tiene la columna '{col}'.")

    df["start_date_time"] = pd.to_datetime(df["start_date_time"], errors="coerce")
    df = df.dropna(subset=["start_date_time"])

    # Fecha de referencia: día siguiente a la última transacción
    ref_date = df["start_date_time"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("user_id")
        .agg(
            recency=("start_date_time", lambda x: (ref_date - x.max()).days),
            frequency=("user_id", "count"),
            monetary=("ingresos_cop", "sum"),
        )
        .reset_index()
    )

    # Recency: menor recency = mejor ⇒ etiquetas altas para valores pequeños
    rfm["R_score"] = _score_from_series(
        rfm["recency"], labels=[4, 3, 2, 1], ascending=True
    ).astype(int)

    # Frequency y Monetary: mayor valor = mejor ⇒ labels crecientes
    rfm["F_score"] = _score_from_series(
        rfm["frequency"], labels=[1, 2, 3, 4], ascending=True
    ).astype(int)

    rfm["M_score"] = _score_from_series(
        rfm["monetary"], labels=[1, 2, 3, 4], ascending=True
    ).astype(int)

    # Score total
    rfm["RFM_score"] = rfm[["R_score", "F_score", "M_score"]].sum(axis=1)

    # Segmento según score
    def asignar_segmento(score):
        if score >= 11:
            return "VIP"
        elif score >= 9:
            return "Leales"
        elif score >= 6:
            return "Potenciales"
        else:
            return "En riesgo"

    rfm["segment"] = rfm["RFM_score"].apply(asignar_segmento)

    return rfm


def crear_grafico_2d(rfm: pd.DataFrame):
    """
    Recibe el DataFrame RFM (ya calculado en dashboard_profesional
    cuando 'rfm': True) y dibuja Frecuencia vs Ingresos.
    """
    data = rfm.copy()
    data["monetary_millones"] = data["monetary"] / 1_000_000

    fig = px.scatter(
        data,
        x="frequency",
        y="monetary_millones",
        color="segment",
        size="monetary_millones",
        hover_data=["user_id", "recency", "RFM_score"],
        labels={
            "frequency": "Frecuencia (nº de transacciones)",
            "monetary_millones": "Ingresos (millones COP)",
            "segment": "Segmento RFM",
        },
        title="Segmentación RFM (Frecuencia vs Ingresos)",
    )

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Frecuencia (nº de transacciones)",
        yaxis_title="Ingresos (millones de COP)",
        margin=dict(l=40, r=20, t=60, b=40),
        legend_title_text="Segmento",
    )
    return fig
