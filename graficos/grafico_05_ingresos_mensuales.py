import pandas as pd
import plotly.express as px


def preparar_datos_ingresos_mensuales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega los ingresos por año-mes a partir del dataframe completo.
    Trabaja siempre sobre la columna 'ingresos_cop' (CREADA en dashboard_profesional).
    """
    df = df.copy()

    if "start_date_time" not in df.columns:
        raise ValueError("El dataset no tiene la columna 'start_date_time'.")

    if "ingresos_cop" not in df.columns:
        raise ValueError("El dataset no tiene la columna 'ingresos_cop'.")

    # Normalizar fecha a año-mes
    df["anio_mes"] = pd.to_datetime(df["start_date_time"], errors="coerce")
    df = df.dropna(subset=["anio_mes"])
    df["anio_mes"] = df["anio_mes"].dt.to_period("M").dt.to_timestamp()

    mensual = (
        df.groupby("anio_mes", as_index=False)["ingresos_cop"]
        .sum()
        .sort_values("anio_mes")
    )

    mensual["ingresos_millones"] = mensual["ingresos_cop"] / 1_000_000
    return mensual


def crear_grafico(df: pd.DataFrame):
    mensual = preparar_datos_ingresos_mensuales(df)

    fig = px.bar(
        mensual,
        x="anio_mes",
        y="ingresos_millones",
        text="ingresos_millones",
        labels={
            "anio_mes": "Mes",
            "ingresos_millones": "Ingresos (millones COP)",
        },
        title="Ingresos mensuales",
    )

    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Mes",
        yaxis_title="Ingresos (millones de COP)",
        xaxis_tickformat="%Y-%m",
        margin=dict(l=40, r=20, t=60, b=40),
        uniformtext_minsize=8,
        uniformtext_mode="hide",
    )
    return fig
