# -*- coding: utf-8 -*-
"""
Gráfico 11: Retención de Usuarios (Cohortes mensuales)
- Cohorte = mes de primera transacción del usuario.
- Retención t = % de usuarios del cohorte activos en el mes t (t=0…N).
- Consola: totales, one-timers, repeaters, 2nd purchase, resumen por cohorte.
- Guarda figura en /outputs/grafico_11_retencion_usuarios.png
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -------------------------------------------------------------
# Import utils desde la raíz del proyecto
# -------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))   # .../graficos
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)                # .../ANALISIS_DATOS
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import cargar_datos, guardar_grafico  # noqa: E402

CSV_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "data", "df_oasis_clean_normalized.csv"),
    os.path.join(PROJECT_ROOT, "data", "df_oasis_normalizado.csv"),
    os.path.join(PROJECT_ROOT, "data", "df_oasis_clean.csv"),
    os.path.join(PROJECT_ROOT, "data", "df_oasis_ready.csv"),
]

# -------------------------------------------------------------
# Carga + normalización (alineado con los otros gráficos)
# -------------------------------------------------------------
def _pick_existing_path(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("No se encontró CSV. Probé:\n- " + "\n- ".join(paths))

def cargar_df() -> pd.DataFrame:
    csv_path = _pick_existing_path(CSV_CANDIDATES)
    df = cargar_datos(csv_path)
    print(f"✓ Usando archivo: {csv_path}")

    # Fecha
    if "start_date_time" not in df.columns:
        raise ValueError("Falta columna 'start_date_time'.")
    df["start_date_time"] = pd.to_datetime(df["start_date_time"], errors="coerce")
    df = df.dropna(subset=["start_date_time"])

    # IDs mínimos
    if "id" not in df.columns:
        if "transaction_id" in df.columns:
            df = df.rename(columns={"transaction_id": "id"})
        else:
            df = df.reset_index().rename(columns={"index": "id"})
    if "user_id" not in df.columns:
        raise ValueError("Falta columna 'user_id'.")

    # Monetario (COP) – para métricas auxiliares
    if "ingresos_cop" in df.columns:
        df["ingresos_cop"] = pd.to_numeric(df["ingresos_cop"], errors="coerce")
    elif "amount_transaction_cop" in df.columns:
        df["ingresos_cop"] = pd.to_numeric(df["amount_transaction_cop"], errors="coerce")
    elif "amount_transaction" in df.columns:
        df["ingresos_cop"] = pd.to_numeric(df["amount_transaction"], errors="coerce") / 100.0
    else:
        df["ingresos_cop"] = np.nan

    return df

# -------------------------------------------------------------
# Cohortes y Retención
# -------------------------------------------------------------
def preparar_cohortes(df: pd.DataFrame):
    d = df.copy()
    d["year_month"] = d["start_date_time"].dt.to_period("M").astype(str)

    # Cohorte = primer mes en que el usuario aparece
    first_tx = (
        d.groupby("user_id")["start_date_time"].min().dt.to_period("M").astype(str)
    )
    d = d.merge(first_tx.rename("cohort_month"), on="user_id", how="left")

    # Mes de actividad (calendar month)
    d["active_month"] = d["start_date_time"].dt.to_period("M").astype(str)

    # Offset = meses desde el mes de cohorte
    # Pasamos a Timestamp para restar y tomar diferencia en meses
    d["_cohort_ts"] = pd.to_datetime(d["cohort_month"])
    d["_active_ts"] = pd.to_datetime(d["active_month"])
    d["period_number"] = (
        (d["_active_ts"].dt.year - d["_cohort_ts"].dt.year) * 12
        + (d["_active_ts"].dt.month - d["_cohort_ts"].dt.month)
    )
    d = d[d["period_number"] >= 0]

    # Usuarios activos por (cohorte, period_number)
    active_users = (
        d.groupby(["cohort_month", "period_number"])["user_id"]
         .nunique()
         .reset_index(name="active_users")
    )

    # Tamaño de cada cohorte (period_number=0)
    cohort_sizes = (
        active_users[active_users["period_number"] == 0]
        .rename(columns={"active_users": "cohort_size"})
        .loc[:, ["cohort_month", "cohort_size"]]
    )

    # Unir tamaños y calcular % retención
    ret = active_users.merge(cohort_sizes, on="cohort_month", how="left")
    ret["retention_pct"] = np.where(
        ret["cohort_size"] > 0, ret["active_users"] / ret["cohort_size"] * 100.0, np.nan
    )

    # Matriz cohort (filas=cohorte, columnas=period_number) con %
    retention_matrix = (
        ret.pivot(index="cohort_month", columns="period_number", values="retention_pct")
           .sort_index()
           .round(1)
    )

    # Curva de retención promedio (promedio por columna)
    avg_curve = retention_matrix.mean(axis=0, skipna=True)

    return retention_matrix, cohort_sizes.sort_values("cohort_month"), avg_curve, d

def calcular_one_timers_repeaters(df: pd.DataFrame):
    g = df.groupby("user_id").size()
    usuarios_unicos = g.shape[0]
    usuarios_una_vez = int(g.eq(1).sum())
    usuarios_repetidores = usuarios_unicos - usuarios_una_vez
    tasa_repeticion = usuarios_repetidores / usuarios_unicos * 100 if usuarios_unicos else 0

    # Tasa de segunda compra: % con tamaño >= 2
    tasa_segunda = usuarios_repetidores / usuarios_unicos * 100 if usuarios_unicos else 0

    return {
        "usuarios_unicos": int(usuarios_unicos),
        "one_timers": int(usuarios_una_vez),
        "repeaters": int(usuarios_repetidores),
        "tasa_repeticion_pct": tasa_repeticion,
        "tasa_segunda_compra_pct": tasa_segunda,
    }

# -------------------------------------------------------------
# Visualización
# -------------------------------------------------------------
def crear_grafico(retention_matrix: pd.DataFrame, avg_curve: pd.Series, stats: dict) -> go.Figure:
    # Preparamos matrices/series (máximo 12 periodos para legibilidad, si hay más se muestran también)
    heat = retention_matrix.copy()

    # Subplots: Heatmap + curva + barras one-timers/repeaters
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "heatmap", "colspan": 2}, None],
               [{"type": "scatter"}, {"type": "bar"}]],
        subplot_titles=(
            "Heatmap de Retención por Cohorte (% de usuarios activos)",
            "Curva de Retención Promedio por Periodo",
            "Usuarios: One-timers vs Repeaters"
        ),
        vertical_spacing=0.13,
        horizontal_spacing=0.12,
        row_heights=[0.62, 0.38]
    )

    # Heatmap
    fig.add_trace(
        go.Heatmap(
            z=heat.values,
            x=[f"Mes {c}" for c in heat.columns],
            y=list(heat.index),
            colorscale="Blues",
            zmin=0, zmax=100,
            colorbar=dict(title="% Retención", x=1.02),
            hovertemplate="Cohorte: %{y}<br>Periodo: %{x}<br>Retención: %{z:.1f}%<extra></extra>"
        ),
        row=1, col=1
    )

    # Curva promedio
    fig.add_trace(
        go.Scatter(
            x=[int(i) for i in avg_curve.index],
            y=avg_curve.values,
            mode="lines+markers",
            name="Retención promedio",
            hovertemplate="Mes %{x}: %{y:.1f}%<extra></extra>"
        ),
        row=2, col=1
    )

    # Barras one-timers vs repeaters
    fig.add_trace(
        go.Bar(
            x=["One-timers (1 visita)", "Repeaters (≥2)"],
            y=[stats["one_timers"], stats["repeaters"]],
            text=[f"{stats['one_timers']:,}", f"{stats['repeaters']:,}"],
            textposition="outside",
            hovertemplate="%{x}: %{y:,} usuarios<extra></extra>"
        ),
        row=2, col=2
    )

    # Layout
    fig.update_layout(
        title=dict(
            text="Retención de Usuarios por Cohortes",
            x=0.5, xanchor="center",
            font=dict(size=22, family="Arial Black", color="#2d3748")
        ),
        height=980,
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=11),
        margin=dict(l=70, r=30, t=80, b=50)
    )
    # Ejes
    fig.update_xaxes(gridcolor="#e2e8f0", row=2, col=1, title_text="Mes desde la primera compra")
    fig.update_yaxes(gridcolor="#e2e8f0", row=2, col=1, title_text="% Retención")
    fig.update_yaxes(gridcolor="#e2e8f0", row=2, col=2, title_text="Usuarios")

    return fig

# -------------------------------------------------------------
# Consola: tablas y checks
# -------------------------------------------------------------
def imprimir_consola(retention_matrix: pd.DataFrame, cohort_sizes: pd.DataFrame, stats: dict, df: pd.DataFrame):
    print("\n" + "="*100)
    print("RESUMEN DE RETENCIÓN")
    print("="*100)

    total_tx = len(df)
    total_cop = df["ingresos_cop"].sum(skipna=True)
    print(f"• Registros totales:       {total_tx:,}")
    print(f"• Usuarios únicos:         {stats['usuarios_unicos']:,}")
    print(f"• One-timers (1 visita):   {stats['one_timers']:,}  ({stats['one_timers']/stats['usuarios_unicos']*100:,.1f}%)")
    print(f"• Repeaters (≥2 visitas):  {stats['repeaters']:,}  ({stats['tasa_repeticion_pct']:,.1f}%)")
    print(f"• Tasa 2ª compra:          {stats['tasa_segunda_compra_pct']:,.1f}%")
    if pd.notna(total_cop):
        print(f"• Ingresos totales (COP):  ${total_cop:,.0f}")

    print("\n" + "-"*100)
    print("TAMAÑO DE COHORTES (primeros 10)")
    print("-"*100)
    print(f"{'Cohorte':<12} {'Usuarios':>10}")
    for _, r in cohort_sizes.head(10).iterrows():
        print(f"{r['cohort_month']:<12} {int(r['cohort_size']):>10,}")

    # Mostrar primeras 6 columnas (mes 0..5) si existen
    cols = [c for c in retention_matrix.columns if c <= 5]
    sub = retention_matrix[cols].copy() if cols else retention_matrix.copy()

    print("\n" + "-"*100)
    print("MATRIZ DE RETENCIÓN (% usuarios activos)  —  Meses 0..5")
    print("-"*100)
    if sub.empty:
        print("(Sin datos suficientes para cohortes)")
    else:
        header = "Cohorte    " + " ".join([f"{c:>8}" for c in sub.columns])
        print(header)
        for idx, row in sub.iterrows():
            vals = " ".join([f"{v:>8.1f}" if not np.isnan(v) else f"{'—':>8}" for v in row.values])
            print(f"{idx:<10} {vals}")

# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "="*100)
    print("GRÁFICO 11: RETENCIÓN DE USUARIOS (COHORTES)")
    print("="*100)

    # 1) Cargar datos
    df = cargar_df()
    print(f"✓ Registros: {len(df):,}")

    # 2) Cohortes y retención
    retention_matrix, cohort_sizes, avg_curve, df_cohorted = preparar_cohortes(df)

    # 3) Métricas de one-timers / repeaters
    stats = calcular_one_timers_repeaters(df)

    # 4) Imprimir tablas en consola
    imprimir_consola(retention_matrix, cohort_sizes, stats, df)

    # 5) Graficar
    fig = crear_grafico(retention_matrix, avg_curve, stats)

    # 6) Guardar/mostrar
    print("\n" + "="*100)
    print("Guardando figura…")
    guardar_grafico(fig, "grafico_11_retencion_usuarios.png")
    print("Abriendo figura…")
    fig.show()
