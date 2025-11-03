# -*- coding: utf-8 -*-
"""
Gráfico 8: Top Estaciones (total coherente ≈ 166M COP)
- Normalización CONSISTENTE:
    * ingresos_cop = amount_transaction / 100   (centavos → COP)
    * energy_kwh: usa energy_kwh; si solo hay energy_wh → /1000; si hay 'energy' aplica heurística
- Consola: totales, % que explica el Top-15, Top-3 y tabla Top-10.
- Gráficas:
    1) Barras Top-N por ingresos (M COP)
    2) Radar comparativo (Transacciones, Usuarios, Energía, Ingresos) para Top-6
- Salida: /outputs/grafico_08_top_estaciones_principal.png
          /outputs/grafico_08_top_estaciones_radar.png
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------------------------------------------------------
# Import utils desde la raíz del proyecto
# ------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))   # .../graficos
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)                # .../analisis_datos
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import cargar_datos, guardar_grafico  # noqa: E402

CSV_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "data", "df_oasis_clean_normalized.csv"),
    os.path.join(PROJECT_ROOT, "data", "df_oasis_normalizado.csv"),
    os.path.join(PROJECT_ROOT, "data", "df_oasis_clean.csv"),
    os.path.join(PROJECT_ROOT, "data", "df_oasis_ready.csv"),
]

# ------------------------------------------------------------------
# Carga + normalización
# ------------------------------------------------------------------
def _pick_existing_path(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("No se encontró CSV. Probé:\n- " + "\n- ".join(paths))

def cargar_df() -> pd.DataFrame:
    csv_path = _pick_existing_path(CSV_CANDIDATES)
    df = cargar_datos(csv_path)
    print(f"✓ Usando archivo: {csv_path}")

    # Fechas
    if "start_date_time" not in df.columns:
        raise ValueError("Falta columna 'start_date_time'.")
    df["start_date_time"] = pd.to_datetime(df["start_date_time"], errors="coerce")
    df = df.dropna(subset=["start_date_time"])

    # ID transacción
    if "id" not in df.columns:
        if "transaction_id" in df.columns:
            df = df.rename(columns={"transaction_id": "id"})
        else:
            df = df.reset_index().rename(columns={"index": "id"})

    # Estación
    if "evse_uid" not in df.columns:
        for cand in ["station_name", "evse_id", "charger_id"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "evse_uid"})
                break
    if "evse_uid" not in df.columns:
        raise ValueError("Falta columna de estación ('evse_uid').")

    # Energía (kWh)
    energy_cols = [c for c in df.columns if c.lower() in ("energy_kwh", "energy_wh", "energy")]
    if not energy_cols:
        df["energy_kwh"] = np.nan
    else:
        col = energy_cols[0]
        s = pd.to_numeric(df[col], errors="coerce")
        if col.lower() == "energy_kwh":
            df["energy_kwh"] = s
        elif col.lower() == "energy_wh":
            df["energy_kwh"] = s / 1000.0
        else:  # 'energy' ambiguo
            maxv = s.max(skipna=True)
            df["energy_kwh"] = s / 1000.0 if (pd.notna(maxv) and maxv > 2000) else s

    # Ingresos (COP) — regla confirmada
    if "ingresos_cop" in df.columns:
        df["ingresos_cop"] = pd.to_numeric(df["ingresos_cop"], errors="coerce")
    elif "amount_transaction_cop" in df.columns:
        df["ingresos_cop"] = pd.to_numeric(df["amount_transaction_cop"], errors="coerce")
    elif "amount_transaction" in df.columns:
        df["ingresos_cop"] = pd.to_numeric(df["amount_transaction"], errors="coerce") / 100.0
    else:
        raise ValueError("No encuentro columna monetaria. Se espera 'amount_transaction'.")

    return df

# ------------------------------------------------------------------
# Métricas por estación
# ------------------------------------------------------------------
def calcular_metricas_estaciones(df: pd.DataFrame) -> pd.DataFrame:
    metricas = (
        df.groupby("evse_uid")
          .agg(
              total_transacciones=("id", "count"),
              usuarios_unicos=("user_id", "nunique"),
              ingresos_totales_cop=("ingresos_cop", "sum"),
              ingreso_promedio_cop=("ingresos_cop", "mean"),
              ingreso_mediano_cop=("ingresos_cop", "median"),
              energia_total_kwh=("energy_kwh", "sum"),
              energia_promedio_kwh=("energy_kwh", "mean"),
              primera_fecha=("start_date_time", "min"),
              ultima_fecha=("start_date_time", "max"),
          )
          .reset_index()
    )

    metricas["dias_activos"] = (metricas["ultima_fecha"] - metricas["primera_fecha"]).dt.days + 1
    metricas["dias_activos"] = metricas["dias_activos"].clip(lower=1)
    metricas["transacciones_por_dia"] = metricas["total_transacciones"] / metricas["dias_activos"]
    metricas["ingresos_por_dia_cop"] = metricas["ingresos_totales_cop"] / metricas["dias_activos"]

    # Conversión para mostrar
    metricas["ingresos_totales_M"] = metricas["ingresos_totales_cop"] / 1_000_000.0
    metricas["ingresos_por_dia_K"] = metricas["ingresos_por_dia_cop"] / 1_000.0

    # Orden principal por ingresos
    metricas = metricas.sort_values("ingresos_totales_cop", ascending=False)
    return metricas

# ------------------------------------------------------------------
# Figuras
# ------------------------------------------------------------------
def figura_barras_top(metricas: pd.DataFrame, top_n: int = 20) -> go.Figure:
    top = metricas.head(top_n).copy()
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=top["evse_uid"],
            x=top["ingresos_totales_M"],
            orientation="h",
            text=[f"${v:,.2f}M" for v in top["ingresos_totales_M"]],
            textposition="outside",
            marker_color="#3b82f6",
            hovertemplate="<b>%{y}</b><br>Ingresos: $%{x:.2f}M COP<extra></extra>",
            name="Ingresos (M COP)"
        )
    )

    fig.update_layout(
        title=dict(
            text="Top Estaciones por Ingresos (M COP)",
            x=0.5, xanchor="center",
            font=dict(size=22, family="Arial Black", color="#2d3748")
        ),
        height=900,
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=11),
        margin=dict(l=70, r=30, t=80, b=40)
    )
    fig.update_xaxes(title_text="Ingresos (Millones de COP)", gridcolor="#e2e8f0")
    fig.update_yaxes(gridcolor="#e2e8f0", tickfont=dict(size=10))
    return fig

def figura_radar(metricas: pd.DataFrame, top_n: int = 6) -> go.Figure:
    top = metricas.head(top_n).copy()

    # Escalas comparables (min-max normalizado por métrica)
    cols = ["total_transacciones", "usuarios_unicos", "energia_total_kwh", "ingresos_totales_cop"]
    norm = top[cols].copy()
    for c in cols:
        cmax = norm[c].max()
        cmin = norm[c].min()
        norm[c] = 0 if cmax == cmin else (norm[c] - cmin) / (cmax - cmin)

    categorias = ["Transacciones", "Usuarios", "Energía (kWh)", "Ingresos (COP)"]
    fig = go.Figure()
    for i, (_, row) in enumerate(norm.iterrows()):
        valores = [row["total_transacciones"], row["usuarios_unicos"],
                   row["energia_total_kwh"], row["ingresos_totales_cop"]]
        fig.add_trace(
            go.Scatterpolar(
                r=valores + [valores[0]],
                theta=categorias + [categorias[0]],
                fill="toself",
                name=top.iloc[i]["evse_uid"]
            )
        )

    fig.update_layout(
        title=dict(
            text="Top 6 estaciones — Perfil comparativo (Transacciones, Usuarios únicos, Energía kWh, Ingresos COP)",
            x=0.5, xanchor="center",
            font=dict(size=20, family="Arial Black", color="#2d3748")
        ),
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=720,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig

# ------------------------------------------------------------------
# Consola
# ------------------------------------------------------------------
def imprimir_resumen_consola(df: pd.DataFrame, metricas: pd.DataFrame):
    total_cop = df["ingresos_cop"].sum(skipna=True)
    top15 = metricas.head(15)
    top15_sum = top15["ingresos_totales_cop"].sum(skipna=True)

    print("\n" + "="*100)
    print("GRÁFICO 8 - VALORES CORRECTOS (normalización consistente)")
    print("="*100)
    print("\n💰 TOTALES (COP):")
    print(f"   • Dataset:  ${total_cop:,.0f}  (~{total_cop/1_000_000:.2f} M)")
    print(f"   • Top 15:   ${top15_sum:,.0f}  (~{top15_sum/1_000_000:.2f} M) "
          f"({(top15_sum/total_cop*100 if total_cop else np.nan):.1f}%)")

    print("\n🏆 TOP 3 POR INGRESOS:")
    for i in range(min(3, len(top15))):
        r = top15.iloc[i]
        print(f"   {i+1}. {r['evse_uid']}: ${r['ingresos_totales_M']:.2f} M")

    # Tabla Top-10
    print("\n" + "-"*100)
    print("TOP 10 – ESTACIONES (ingresos, transacciones, usuarios, energía)")
    print("-"*100)
    print(f"{'Estación':<32} {'Ingresos(M)':>12} {'Trans':>8} {'Usuarios':>9} {'kWh':>12} {'Trans/día':>10}")
    for _, r in metricas.head(10).iterrows():
        print(f"{str(r['evse_uid'])[:32]:<32} {r['ingresos_totales_M']:>12.2f} "
              f"{int(r['total_transacciones']):>8} {int(r['usuarios_unicos']):>9} "
              f"{(r['energia_total_kwh'] if pd.notna(r['energia_total_kwh']) else 0):>12,.0f} "
              f"{r['transacciones_por_dia']:>10.2f}")

# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "="*100)
    print("GRÁFICO 8: TOP ESTACIONES")
    print("="*100)

    # 1) Cargar y normalizar
    df = cargar_df()
    print(f"✓ Registros: {len(df):,}")

    # 2) Métricas por estación
    metricas = calcular_metricas_estaciones(df)

    # 3) Imprimir resumen en consola (reconciliación ≈ 166M COP)
    imprimir_resumen_consola(df, metricas)

    # 4) Figuras
    fig_main = figura_barras_top(metricas, top_n=20)
    fig_radar = figura_radar(metricas, top_n=6)

    # 5) Guardar y mostrar
    guardar_grafico(fig_main, "grafico_08_top_estaciones_principal.png")
    guardar_grafico(fig_radar, "grafico_08_top_estaciones_radar.png")

    print("🌐 Abriendo figuras…")
    fig_main.show()
    fig_radar.show()
