import pandas as pd
import plotly.express as px


def preparar_datos_temporales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve un dataframe con columnas:
    - start_date_time (normalizada)
    - hora (0–23)
    - dia_semana (0=lunes, 6=domingo)
    """
    df = df.copy()

    if "start_date_time" not in df.columns:
        raise ValueError("El dataset no tiene la columna 'start_date_time'.")

    df["start_date_time"] = pd.to_datetime(df["start_date_time"], errors="coerce")
    df = df.dropna(subset=["start_date_time"])

    df["hora"] = df["start_date_time"].dt.hour
    df["dia_semana"] = df["start_date_time"].dt.dayofweek

    return df


def crear_matriz_heatmap(df_t: pd.DataFrame) -> pd.DataFrame:
    """
    Crea una matriz (día de la semana x hora) con el número de transacciones.
    Aunque aquí la usemos poco, el dashboard la espera.
    """
    if "user_id" not in df_t.columns:
        # si faltara user_id, contamos filas
        df_t = df_t.copy()
        df_t["user_id"] = 1

    matriz = df_t.pivot_table(
        index="dia_semana",
        columns="hora",
        values="user_id",
        aggfunc="count",
        fill_value=0,
    )
    return matriz


def crear_grafico(df_t: pd.DataFrame, matriz: pd.DataFrame):
    """
    Recibe df temporal y la matriz (por compatibilidad con dashboard_profesional)
    y muestra un gráfico de líneas de transacciones por hora.
    """
    counts = (
        df_t.groupby("hora")
        .size()
        .rename("transacciones")
        .reset_index()
        .sort_values("hora")
    )

    fig = px.line(
        counts,
        x="hora",
        y="transacciones",
        markers=True,
        labels={
            "hora": "Hora del día",
            "transacciones": "Número de transacciones",
        },
        title="Patrones horarios de uso",
    )

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Hora del día",
        yaxis_title="Transacciones",
        xaxis=dict(dtick=1),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig
