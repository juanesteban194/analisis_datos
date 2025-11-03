import pandas as pd
import plotly.express as px
import grafico_03_segmentacion_rfm as rfm_mod


def _clv_por_segmento(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula, para cada segmento RFM:
      - número de usuarios
      - ingreso total
      - CLV promedio (ingreso_promedio) en COP
    """
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

    # Solo reordenamos los segmentos para que salgan bonitos en el eje X
    orden = ["VIP", "Leales", "Potenciales", "En riesgo"]
    seg["segment"] = pd.Categorical(seg["segment"], categories=orden, ordered=True)
    seg = seg.sort_values("segment")

    # Imprimir tabla en consola para que puedas verificar los valores reales
    print("\n===== CLV POR SEGMENTO (COP) =====")
    print(seg[["segment", "usuarios", "ingreso_total", "ingreso_promedio"]])
    print("=================================\n")

    return seg


def crear_grafico(df: pd.DataFrame):
    """
    Dibuja el CLV promedio por segmento en COP (no en millones),
    para que los valores se vean tal cual.
    """
    seg = _clv_por_segmento(df)

    # Texto formateado con separador de miles
    textos = seg["ingreso_promedio"].apply(lambda v: f"${v:,.0f} COP")

    fig = px.bar(
        seg,
        x="segment",
        y="ingreso_promedio",
        text=textos,
        labels={
            "segment": "Segmento",
            "ingreso_promedio": "CLV promedio (COP)",
        },
        title="CLV promedio por segmento RFM",
    )

    fig.update_traces(textposition="outside")

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Segmento RFM",
        yaxis_title="CLV promedio (COP)",
        margin=dict(l=40, r=20, t=60, b=40),
    )

    return fig
