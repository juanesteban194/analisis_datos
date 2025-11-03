import pandas as pd
import plotly.express as px
import grafico_03_segmentacion_rfm as rfm_mod


def _clv_por_segmento(df: pd.DataFrame) -> pd.DataFrame:
    rfm = rfm_mod.calcular_rfm(df)

    seg = (
        rfm.groupby("segment")
        .agg(
            usuarios=("user_id", "nunique"),
            ingreso_total=("monetary", "sum"),
            ingreso_promedio=("monetary", "mean"),
        )
        .reset_index()
    )

    seg["clv_promedio_millones"] = seg["ingreso_promedio"] / 1_000_000
    seg["ingreso_total_millones"] = seg["ingreso_total"] / 1_000_000

    orden = ["VIP", "Leales", "Potenciales", "En riesgo"]
    seg["segment"] = pd.Categorical(seg["segment"], categories=orden, ordered=True)
    seg = seg.sort_values("segment")

    return seg


def crear_grafico(df: pd.DataFrame):
    seg = _clv_por_segmento(df)

    fig = px.bar(
        seg,
        x="segment",
        y="clv_promedio_millones",
        text="clv_promedio_millones",
        labels={
            "segment": "Segmento",
            "clv_promedio_millones": "CLV promedio (millones COP)",
        },
        title="CLV promedio por segmento RFM",
    )

    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Segmento RFM",
        yaxis_title="CLV promedio (millones de COP)",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig
