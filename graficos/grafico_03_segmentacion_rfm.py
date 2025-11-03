import pandas as pd
import plotly.express as px


def calcular_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula Recency, Frequency, Monetary y el score RFM por usuario,
    usando 'ingresos_cop' como valor monetario.
    """
    df = df.copy()

    for col in ["user_id", "start_date_time", "ingresos_cop"]:
        if col not in df.columns:
            raise ValueError(f"El dataset no tiene la columna '{col}'.")

    df["start_date_time"] = pd.to_datetime(df["start_date_time"], errors="coerce")
    df = df.dropna(subset=["start_date_time"])

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

    def safe_qcut(series, q, labels):
        series = series.copy()
        n_unique = series.nunique()
        if n_unique < q:
            ranks = pd.qcut(
                series.rank(method="first"),
                q=q,
                labels=labels,
                duplicates="drop",
            )
            return ranks
        return pd.qcut(series, q=q, labels=labels, duplicates="drop")

    rfm["R_score"] = safe_qcut(rfm["recency"], 4, labels=[4, 3, 2, 1]).astype(int)
    rfm["F_score"] = safe_qcut(rfm["frequency"], 4, labels=[1, 2, 3, 4]).astype(int)
    rfm["M_score"] = safe_qcut(rfm["monetary"], 4, labels=[1, 2, 3, 4]).astype(int)

    rfm["RFM_score"] = rfm[["R_score", "F_score", "M_score"]].sum(axis=1)

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
    Recibe directamente el dataframe RFM (lo pasa dashboard_profesional
    cuando 'rfm': True) y dibuja frecuencia vs. valor monetario.
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
