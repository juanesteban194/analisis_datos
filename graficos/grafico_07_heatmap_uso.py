# -*- coding: utf-8 -*-
"""
Gráfico 7: Heatmap de Uso de Estaciones
Análisis: Uso detallado por estación con comparación temporal
Tipo: Heatmap de estaciones vs tiempo + análisis comparativo

⚙️ Normalización monetaria:
- Base en COP (columna garantizada 'ingresos_cop').
- Si no existe, se toma 'amount_transaction_cop'; si no, 'amount_transaction'/100 (centavos→COP).
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
import os
import sys

# --------------------------------------------------------------------
# Agregar el directorio raíz al path (ANALISIS_DATOS/)
# --------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))   # .../graficos
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)                # .../ANALISIS_DATOS
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import cargar_datos, guardar_grafico  # usa tu utils.py de la raíz


# --------------------------------------------------------------------
# CARGA ROBUSTA + NORMALIZACIÓN A COP
# --------------------------------------------------------------------
CSV_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "data", "df_oasis_normalizado.csv"),
    os.path.join(PROJECT_ROOT, "data", "df_oasis_clean_normalized.csv"),
    os.path.join(PROJECT_ROOT, "data", "df_oasis_clean.csv"),
    os.path.join(PROJECT_ROOT, "data", "df_oasis_ready.csv"),
]

def _pick_existing_path(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("No encontré CSV. Probé:\n- " + "\n- ".join(paths))

def cargar_df() -> pd.DataFrame:
    csv_path = _pick_existing_path(CSV_CANDIDATES)
    df = cargar_datos(csv_path)
    print(f"✓ Usando archivo: {csv_path}")

    # Tipos
    if "start_date_time" not in df.columns:
        raise ValueError("Falta columna 'start_date_time'.")
    df["start_date_time"] = pd.to_datetime(df["start_date_time"], errors="coerce")
    df = df.dropna(subset=["start_date_time"])

    # id de transacción (por si no existe 'id')
    if "id" not in df.columns:
        if "transaction_id" in df.columns:
            df = df.rename(columns={"transaction_id": "id"})
        else:
            # crea un id sintético estable
            df = df.reset_index().rename(columns={"index": "id"})
    # usuario
    if "user_id" not in df.columns:
        raise ValueError("Falta columna 'user_id'.")

    # ingresos en COP
    if "ingresos_cop" in df.columns:
        df["ingresos_cop"] = pd.to_numeric(df["ingresos_cop"], errors="coerce")
    elif "amount_transaction_cop" in df.columns:
        df["ingresos_cop"] = pd.to_numeric(df["amount_transaction_cop"], errors="coerce")
    elif "amount_transaction" in df.columns:
        df["ingresos_cop"] = pd.to_numeric(df["amount_transaction"], errors="coerce") / 100.0
    else:
        raise ValueError(
            "No encontré columna monetaria. Se requiere 'ingresos_cop' o "
            "'amount_transaction_cop' o 'amount_transaction' (centavos)."
        )

    # energía
    if "energy_kwh" in df.columns:
        df["energy_kwh"] = pd.to_numeric(df["energy_kwh"], errors="coerce")
    else:
        df["energy_kwh"] = np.nan  # si no existe, seguimos sin romper

    # estación
    if "evse_uid" not in df.columns:
        # intenta otra columna común (ajusta si tienes un nombre distinto)
        for cand in ["station_name", "evse_id", "charger_id"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "evse_uid"})
                break
    if "evse_uid" not in df.columns:
        raise ValueError("Falta columna de estación ('evse_uid').")

    return df


# --------------------------------------------------------------------
# PREPARACIÓN DE DATOS DE USO (sumas en COP reales)
# --------------------------------------------------------------------
def preparar_datos_uso(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega métricas por estación con ingresos en COP (coherente con total ≈ 166M).
    """
    uso_por_estacion = df.groupby("evse_uid").agg(
        transacciones=("id", "count"),
        usuarios_unicos=("user_id", "nunique"),
        ingresos_totales=("ingresos_cop", "sum"),
        ingreso_promedio=("ingresos_cop", "mean"),
        energia_total=("energy_kwh", "sum"),
        energia_promedio=("energy_kwh", "mean"),
        primera_transaccion=("start_date_time", "min"),
        ultima_transaccion=("start_date_time", "max"),
    ).reset_index()

    # Días activos y tasas
    dias = (uso_por_estacion["ultima_transaccion"] - uso_por_estacion["primera_transaccion"]).dt.days + 1
    dias = dias.replace({0: 1})  # evita división por cero
    uso_por_estacion["dias_activo"] = dias

    uso_por_estacion["trans_por_dia"] = uso_por_estacion["transacciones"] / uso_por_estacion["dias_activo"]
    uso_por_estacion["ingresos_por_dia"] = uso_por_estacion["ingresos_totales"] / uso_por_estacion["dias_activo"]

    # Tipo de estación (heurística simple por nombre/uid)
    def identificar_tipo(nombre: str) -> str:
        n = str(nombre).lower()
        if "éxito" in n or "exito" in n:
            return "Éxito"
        if "ccs" in n:
            return "CCS"
        if "t1" in n or "t2" in n:
            return "Tipo 1/2"
        return "Otro"

    uso_por_estacion["tipo"] = uso_por_estacion["evse_uid"].apply(identificar_tipo)

    # Orden
    uso_por_estacion = uso_por_estacion.sort_values("transacciones", ascending=False)
    return uso_por_estacion


# --------------------------------------------------------------------
# MATRIZ TEMPORAL (transacciones por estación y mes)
# --------------------------------------------------------------------
def crear_matriz_temporal(df: pd.DataFrame, orden_evse: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["year_month"] = df["start_date_time"].dt.to_period("M").astype(str)

    matriz = (
        df.groupby(["evse_uid", "year_month"]).size().reset_index(name="transacciones")
    )
    pivot = matriz.pivot(index="evse_uid", columns="year_month", values="transacciones").fillna(0)

    # Reordenar filas al orden de top estaciones
    pivot = pivot.reindex(orden_evse)
    return pivot


# --------------------------------------------------------------------
# GRÁFICOS
# --------------------------------------------------------------------
def crear_grafico(uso_df: pd.DataFrame, matriz_temporal: pd.DataFrame) -> go.Figure:
    # Top 20 para visualización
    top_20 = uso_df.head(20)
    matriz_top = matriz_temporal.loc[top_20["evse_uid"]]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Heatmap: Uso de Estaciones en el Tiempo (Top 20)",
            "Top 20 Estaciones por Transacciones",
            "Ingresos por Estación (Top 20)",
            "Usuarios Únicos por Estación (Top 20)",
        ),
        specs=[[{"type": "heatmap", "colspan": 2}, None],
               [{"type": "bar"}, {"type": "bar"}]],
        row_heights=[0.5, 0.5],
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )

    # Heatmap temporal (transacciones)
    fig.add_trace(
        go.Heatmap(
            z=matriz_top.values,
            x=matriz_top.columns,
            y=matriz_top.index,
            colorscale="Viridis",
            hovertemplate="<b>%{y}</b><br>Mes: %{x}<br>Transacciones: %{z}<extra></extra>",
            colorbar=dict(title="Transacciones", x=1.02),
            zmin=0
        ),
        row=1, col=1
    )

    # Top 20 por transacciones
    colors_trans = px.colors.sequential.Blues_r[:min(20, len(top_20))]
    fig.add_trace(
        go.Bar(
            y=top_20["evse_uid"],
            x=top_20["transacciones"],
            orientation="h",
            marker_color=colors_trans,
            text=top_20["transacciones"],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Transacciones: %{x}<extra></extra>",
        ),
        row=2, col=1
    )

    # Ingresos por estación (COP), etiqueta en millones para legibilidad
    fig.add_trace(
        go.Bar(
            y=top_20["evse_uid"],
            x=top_20["ingresos_totales"],
            orientation="h",
            marker_color="#10b981",
            text=[f"${v/1e6:.1f} M" for v in top_20["ingresos_totales"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Ingresos: $%{x:,.0f} COP<extra></extra>",
        ),
        row=2, col=2
    )

    fig.update_layout(
        title=dict(text="Análisis de Uso por Estación", x=0.5, xanchor="center",
                   font=dict(size=20, color="#2d3748", family="Arial Black")),
        showlegend=False,
        height=1000,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=10),
        margin=dict(l=80, r=40, t=80, b=60)
    )

    # Ejes
    fig.update_xaxes(title_text="Mes", tickangle=-45, row=1, col=1)
    fig.update_yaxes(title_text="Estación", tickfont={"size": 9}, row=1, col=1)

    fig.update_xaxes(title_text="Transacciones", gridcolor="#e2e8f0", row=2, col=1)
    fig.update_yaxes(tickfont={"size": 9}, row=2, col=1)

    fig.update_xaxes(title_text="Ingresos (COP)", gridcolor="#e2e8f0", row=2, col=2)
    fig.update_yaxes(tickfont={"size": 9}, row=2, col=2)

    return fig


def crear_grafico_comparativo(uso_df: pd.DataFrame) -> go.Figure:
    color_map = {"Éxito": "#10b981", "CCS": "#3b82f6", "Tipo 1/2": "#8b5cf6", "Otro": "#f59e0b"}

    fig = px.scatter(
        uso_df,
        x="transacciones",
        y="ingresos_totales",
        size="usuarios_unicos",
        color="tipo",
        color_discrete_map=color_map,
        hover_data=["evse_uid", "trans_por_dia", "energia_total"],
        labels={
            "transacciones": "Número de Transacciones",
            "ingresos_totales": "Ingresos Totales (COP)",
            "tipo": "Tipo de Estación",
            "usuarios_unicos": "Usuarios Únicos",
        },
        title="Transacciones vs Ingresos por Estación (tamaño = usuarios)"
    )

    fig.update_layout(
        title=dict(x=0.5, xanchor="center",
                   font=dict(size=18, color="#2d3748", family="Arial Black")),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=600,
        font=dict(family="Arial", size=12)
    )
    fig.update_xaxes(gridcolor="#e2e8f0")
    fig.update_yaxes(gridcolor="#e2e8f0")
    return fig


def crear_grafico_eficiencia(uso_df: pd.DataFrame) -> go.Figure:
    top_eficiencia = uso_df.nlargest(20, "trans_por_dia")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top_eficiencia["evse_uid"],
        y=top_eficiencia["trans_por_dia"],
        marker=dict(
            color=top_eficiencia["trans_por_dia"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Trans/Día")
        ),
        text=[f"{v:.1f}" for v in top_eficiencia["trans_por_dia"]],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Transacciones/día: %{y:.2f}<extra></extra>"
    ))
    fig.update_layout(
        title=dict(text="Top 20 Estaciones por Eficiencia (Transacciones por Día)",
                   x=0.5, xanchor="center",
                   font=dict(size=18, color="#2d3748", family="Arial Black")),
        xaxis_title="Estación",
        yaxis_title="Transacciones por Día",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=600,
        font=dict(family="Arial", size=11)
    )
    fig.update_xaxes(tickangle=-45, gridcolor="#e2e8f0")
    fig.update_yaxes(gridcolor="#e2e8f0")
    return fig


# --------------------------------------------------------------------
# ANÁLISIS E INSIGHTS (coherentes con totales)
# --------------------------------------------------------------------
def analizar(uso_df: pd.DataFrame, df: pd.DataFrame) -> dict:
    total_estaciones = len(uso_df)
    total_trans = uso_df["transacciones"].sum()
    promedio_trans = uso_df["transacciones"].mean()
    mediana_trans = uso_df["transacciones"].median()

    top_3 = uso_df.head(3)
    concentracion_top3 = float(top_3["transacciones"].sum()) / float(total_trans) * 100 if total_trans else 0.0

    # Eficiencia
    top_eficiencia = uso_df.nlargest(1, "trans_por_dia")

    # Por tipo
    stats_por_tipo = (
        uso_df.groupby("tipo")
        .agg(
            estaciones=("evse_uid", "count"),
            trans_total=("transacciones", "sum"),
            trans_promedio=("transacciones", "mean"),
            ingresos_total=("ingresos_totales", "sum"),
        )
        .round(2)
    )

    # Distribución de ingresos (share por estación)
    total_ingresos = uso_df["ingresos_totales"].sum()
    share_top = (top_3["ingresos_totales"].sum() / total_ingresos * 100) if total_ingresos else 0.0

    insights = {
        "total_estaciones": int(total_estaciones),
        "total_transacciones": int(total_trans),
        "promedio_trans": float(promedio_trans),
        "mediana_trans": float(mediana_trans),

        "top_1_nombre": str(top_3.iloc[0]["evse_uid"]) if len(top_3) > 0 else "",
        "top_1_trans": int(top_3.iloc[0]["transacciones"]) if len(top_3) > 0 else 0,
        "top_1_ingresos": int(top_3.iloc[0]["ingresos_totales"]) if len(top_3) > 0 else 0,

        "top_2_nombre": str(top_3.iloc[1]["evse_uid"]) if len(top_3) > 1 else "",
        "top_2_trans": int(top_3.iloc[1]["transacciones"]) if len(top_3) > 1 else 0,

        "top_3_nombre": str(top_3.iloc[2]["evse_uid"]) if len(top_3) > 2 else "",
        "top_3_trans": int(top_3.iloc[2]["transacciones"]) if len(top_3) > 2 else 0,

        "concentracion_top3": float(concentracion_top3),

        "bajo_rendimiento_count": int((uso_df["transacciones"] < 50).sum()),
        "bajo_rendimiento_pct": float((uso_df["transacciones"] < 50).mean() * 100),

        "top_eficiencia_nombre": str(top_eficiencia.iloc[0]["evse_uid"]) if len(top_eficiencia) else "",
        "top_eficiencia_valor": float(top_eficiencia.iloc[0]["trans_por_dia"]) if len(top_eficiencia) else 0.0,

        "stats_por_tipo": stats_por_tipo,

        "insight_top": (
            f"La estación líder es '{top_3.iloc[0]['evse_uid']}' con "
            f"{int(top_3.iloc[0]['transacciones'])} transacciones y "
            f"${int(top_3.iloc[0]['ingresos_totales']):,} COP en ingresos."
        ) if len(top_3) else "No hay datos para top 1.",

        "insight_concentracion": (
            f"Las top 3 estaciones ({', '.join(top_3['evse_uid'].astype(str).tolist())}) "
            f"concentran el {concentracion_top3:.1f}% de todas las transacciones."
        ) if len(top_3) >= 3 else "No hay suficientes estaciones para top 3.",

        "insight_bajo_rendimiento": (
            f"{int((uso_df['transacciones'] < 50).sum())} estaciones "
            f"({(uso_df['transacciones'] < 50).mean() * 100:.1f}%) tienen menos de 50 transacciones."
        ),

        "insight_eficiencia": (
            f"La estación más eficiente es '{top_eficiencia.iloc[0]['evse_uid']}' "
            f"con {top_eficiencia.iloc[0]['trans_por_dia']:.1f} transacciones por día."
        ) if len(top_eficiencia) else "No se pudo determinar estación eficiente.",

        "insight_distribucion": f"Las top 3 concentran el {share_top:.1f}% de los ingresos.",
        "total_ingresos_cop": float(total_ingresos),
    }
    return insights


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "="*80)
    print("GRÁFICO 7: HEATMAP DE USO DE ESTACIONES")
    print("="*80)

    # Cargar datos (COP coherente con total ≈ 166M)
    df = cargar_df()
    print(f"✓ Datos cargados: {len(df):,} registros")
    print(f" Total ingresos (COP): {df['ingresos_cop'].sum():,.0f}\n")

    # Preparar datos de uso
    print(" Analizando uso por estación...")
    uso_df = preparar_datos_uso(df)
    print(f"✓ Análisis completado para {len(uso_df)} estaciones\n")

    # Matriz temporal (orden según top estaciones)
    print("  Creando matriz temporal...")
    matriz_temporal = crear_matriz_temporal(df, uso_df["evse_uid"].tolist())
    print("✓ Matriz temporal creada\n")

    # Gráficos
    print(" Creando visualizaciones...")
    fig_main = crear_grafico(uso_df, matriz_temporal)
    fig_comp = crear_grafico_comparativo(uso_df)
    fig_efic = crear_grafico_eficiencia(uso_df)
    print("✓ Gráficos creados\n")

    # Insights
    print("="*80)
    print(" ANÁLISIS DE USO POR ESTACIÓN")
    print("="*80)

    insights = analizar(uso_df, df)

    print(f"\n RESUMEN GENERAL:")
    print(f"   • Total de estaciones: {insights['total_estaciones']}")
    print(f"   • Total de transacciones: {insights['total_transacciones']:,}")
    print(f"   • Promedio por estación: {insights['promedio_trans']:.1f}")
    print(f"   • Mediana: {insights['mediana_trans']:.0f}")

    print(f"\n TOP 3 ESTACIONES:")
    if insights['top_1_nombre']:
        print(f"   1. {insights['top_1_nombre']}: {insights['top_1_trans']:,} trans - ${insights['top_1_ingresos']:,} COP")
    if insights['top_2_nombre']:
        print(f"   2. {insights['top_2_nombre']}: {insights['top_2_trans']:,} trans")
    if insights['top_3_nombre']:
        print(f"   3. {insights['top_3_nombre']}: {insights['top_3_trans']:,} trans")
    print(f"   • Concentración (transacciones): {insights['concentracion_top3']:.1f}%")

    print(f"\n⚡ EFICIENCIA:")
    print(f"   • Estación más eficiente: {insights['top_eficiencia_nombre']}")
    print(f"   • Transacciones por día: {insights['top_eficiencia_valor']:.2f}")

    print(f"\n BAJO RENDIMIENTO:")
    print(f"   • Estaciones con <50 trans: {insights['bajo_rendimiento_count']}")
    print(f"   • Porcentaje: {insights['bajo_rendimiento_pct']:.1f}%")

    print(f"\n ESTADÍSTICAS POR TIPO:")
    print(f"\n{'Tipo':<15} {'Estaciones':>12} {'Trans Total':>15} {'Trans Promedio':>17} {'Ingresos Total':>18}")
    print("-" * 85)
    for tipo, row in insights["stats_por_tipo"].iterrows():
        print(f"{tipo:<15} {int(row['estaciones']):>12} {int(row['trans_total']):>15,} "
              f"{row['trans_promedio']:>17.1f} ${int(row['ingresos_total']):>17,}")

    print(f"\n INSIGHTS PRINCIPALES:")
    print(f"   1) {insights['insight_top']}")
    print(f"   2) {insights['insight_concentracion']}")
    print(f"   3) {insights['insight_bajo_rendimiento']}")
    print(f"   4) {insights['insight_eficiencia']}")
    print(f"   5) {insights['insight_distribucion']}")
    print(f"\n Reconciliación de ingresos (COP): {insights['total_ingresos_cop']:,.0f}")

    # Guardar
    print("\n" + "="*80)
    print(" Guardando gráficos...")
    guardar_grafico(fig_main, "grafico_07_heatmap_uso.png")
    guardar_grafico(fig_comp, "grafico_07_comparativo.png")
    guardar_grafico(fig_efic, "grafico_07_eficiencia.png")

    # Mostrar
    print(" Abriendo gráfico principal en navegador...")
    fig_main.show()
    print(" Abriendo gráfico comparativo en navegador...")
    fig_comp.show()
    print(" Abriendo gráfico de eficiencia en navegador...")
    fig_efic.show()
