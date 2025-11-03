import pandas as pd
import plotly.express as px


def preparar_distribucion_usuarios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Segmenta usuarios por número de transacciones:
    1, 2-5, 6-10, 11+.
    """
    df = df.copy()

    if "user_id" not in df.columns:
        raise ValueError("El dataset no tiene la columna 'user_id'.")

    if "start_date_time" in df.columns:
        df = df.dropna(subset=["start_date_time"])

    tx_por_usuario = (
        df.groupby("user_id")
        .size()
        .rename("num_tx")
        .reset_index()
    )

    def clasificar(n):
        if n == 1:
            return "1 transacción"
        elif 2 <= n <= 5:
            return "2-5 transacciones"
        elif 6 <= n <= 10:
            return "6-10 transacciones"
        else:
            return "11+ transacciones"

    tx_por_usuario["segmento_tx"] = tx_por_usuario["num_tx"].apply(clasificar)

    resumen = (
        tx_por_usuario.groupby("segmento_tx", as_index=False)["user_id"]
        .count()
        .rename(columns={"user_id": "usuarios"})
    )

    total_usuarios = resumen["usuarios"].sum()
    resumen["porcentaje"] = (resumen["usuarios"] / total_usuarios * 100).round(1)

    orden = [
        "1 transacción",
        "2-5 transacciones",
        "6-10 transacciones",
        "11+ transacciones",
    ]
    resumen["segmento_tx"] = pd.Categorical(
        resumen["segmento_tx"], categories=orden, ordered=True
    )
    resumen = resumen.sort_values("segmento_tx")

    return resumen


def crear_grafico(df: pd.DataFrame):
    dist = preparar_distribucion_usuarios(df)

    fig = px.bar(
        dist,
        x="segmento_tx",
        y="usuarios",
        text="usuarios",
        labels={
            "segmento_tx": "Segmento según nº de transacciones",
            "usuarios": "Número de usuarios",
        },
        title="Distribución de usuarios por número de transacciones",
    )

    fig.update_traces(textposition="outside")
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Segmento de usuarios",
        yaxis_title="Usuarios",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig
