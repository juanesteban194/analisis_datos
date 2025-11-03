# -*- coding: utf-8 -*-
"""
Gráfico 10: Energía por Estaciones
Objetivo:
- Medir y comparar consumo de energía (kWh) por estación.
- Mostrar eficiencia (kWh/transacción) y COP/kWh cuando haya ingresos.
- Entregar tablas-resumen en consola y guardar figuras en /outputs.

Normalización:
- Fecha en 'start_date_time' (datetime).
- Estación en 'evse_uid'.
- Transacción en 'id' (o 'transaction_id').
- Energía en kWh en columna 'energy_kwh':
    * Si existe 'energy_kwh' → se usa tal cual (numérico).
    * Si existe 'energy_wh'   → energy_kwh = energy_wh / 1000.
    * Si solo existe 'energy' → heurística:
         - si max(energy) > 2000 → asumimos Wh → /1000
         - en otro caso asumimos ya kWh.
- Ingresos en COP en 'ingresos_cop':
    * Si existe 'ingresos_cop' → usar.
    * Si existe 'amount_transaction_cop' → usar.
    * Si existe 'amount_transaction' → dividir entre 100 (centavos → COP).
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

    # Id transacción
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
        # Si no hay energía, la dejamos como NaN (permite ver métricas monetarias)
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
            if pd.notna(maxv) and maxv > 2000:
                df["energy_kwh"] = s / 1000.0  # era Wh
            else:
                df["energy_kwh"] = s  # ya estaba en kWh

    # Ingresos (COP)
    if "ingresos_cop" in df.columns:
        df["ingresos_cop"] = pd.to_numeric(df["ingresos_cop"], errors="coerce")
    elif "amount_transaction_cop" in df.columns:
        df["ingresos_cop"] = pd.to_numeric(df["amount_transaction_cop"], errors="coerce")
    elif "amount_transaction" in df.columns:
        df["ingresos_cop"] = pd.to_numeric(df["amount_transaction"], errors="coerce") / 100.0
    else:
        df["ingresos_cop"] = np.nan  # permitido: habrá métricas de energía aunque no haya ingresos

    return df

# Clasificador simple (reutilizable)
def clasificar_conector(nombre: str) -> str:
    n = str(nombre).lower()
    if "_t1" in n or " t1" in n or "t101" in n:
        return "T1"
    if "_t2" in n or " t2" in n or "t201" in n or "t202" in n:
        return "T2"
    if "ccs" in n:
        return "CCS"
    return "Otro"

# ------------------------------------------------------------------
# Agregaciones de energía
# ------------------------------------------------------------------
def preparar_metricas_estacion(df: pd.DataFrame) -> pd.DataFrame:
    """Métricas por estación: transacciones, usuarios, kWh, COP/kWh, kWh/día, etc."""
    d = df.copy()
    d["tipo_conector"] = d["evse_uid"].apply(clasificar_conector)

    agg = (
        d.groupby("evse_uid")
         .agg(
             transacciones=("id", "count"),
             usuarios_unicos=("user_id", "nunique"),
             energia_kwh=("energy_kwh", "sum"),
             kwh_prom=("energy_kwh", "mean"),
             ingresos_totales=("ingresos_cop", "sum"),
             first_dt=("start_date_time", "min"),
             last_dt=("start_date_time", "max"),
             tipo_conector=("tipo_conector", "first"),
         )
         .reset_index()
    )

    # días activo y tasas
    agg["dias_activo"] = (agg["last_dt"] - agg["first_dt"]).dt.days + 1
    agg["dias_activo"] = agg["dias_activo"].clip(lower=1)  # evitar divisiones por 0

    agg["kwh_por_dia"] = agg["energia_kwh"] / agg["dias_activo"]
    agg["kwh_por_trans"] = np.where(agg["transacciones"] > 0,
                                    agg["energia_kwh"] / agg["transacciones"],
                                    np.nan)

    # COP por kWh (si hay ingresos y energía)
    agg["cop_por_kwh"] = np.where(agg["energia_kwh"] > 0,
                                  agg["ingresos_totales"] / agg["energia_kwh"],
                                  np.nan)

    # Orden por energía
    agg = agg.sort_values("energia_kwh", ascending=False)

    return agg

def crear_matriz_temporal_kwh(df: pd.DataFrame) -> pd.DataFrame:
    """Heatmap de kWh por estación y por mes (top estaciones)."""
    d = df.copy()
    d["year_month"] = d["start_date_time"].dt.to_period("M").astype(str)
    mat = (
        d.groupby(["evse_uid", "year_month"])["energy_kwh"]
         .sum()
         .reset_index()
    )
    pv = mat.pivot(index="evse_uid", columns="year_month", values="energy_kwh").fillna(0.0)
    # Orden por total
    pv["total"] = pv.sum(axis=1)
    pv = pv.sort_values("total", ascending=False).drop(columns=["total"])
    return pv

# ------------------------------------------------------------------
# Visualización
# ------------------------------------------------------------------
def crear_grafico(agg_est: pd.DataFrame, matriz_top: pd.DataFrame) -> go.Figure:
    """Dashboard de energía con heatmap + barras de métricas."""
    # Tomamos TOP 20 estaciones por kWh para visualización
    top20 = agg_est.head(20)
    heat_top = matriz_top.loc[top20["evse_uid"]] if len(matriz_top) else matriz_top

    fig = make_subplots(
        rows=3, cols=2,
        specs=[
            [{"type": "heatmap", "colspan": 2}, None],
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "bar"}],
        ],
        subplot_titles=(
            "Heatmap: Energía (kWh) por Estación y Mes (Top 20)",
            "Top 20 Estaciones por Energía (kWh)",
            "COP por kWh (Top 20)",
            "kWh por Transacción (Top 20)",
            "kWh por Día (Top 20)"
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.12,
        row_heights=[0.45, 0.30, 0.25]
    )

    # Heatmap kWh
    if len(heat_top):
        fig.add_trace(
            go.Heatmap(
                z=heat_top.values,
                x=list(heat_top.columns),
                y=list(heat_top.index),
                colorscale="Viridis",
                hovertemplate="<b>%{y}</b><br>Mes: %{x}<br>kWh: %{z:,.1f}<extra></extra>",
                colorbar=dict(title="kWh", x=1.02),
            ),
            row=1, col=1
        )

    # Barras: kWh total
    fig.add_trace(
        go.Bar(
            y=top20["evse_uid"],
            x=top20["energia_kwh"],
            orientation="h",
            marker_color="#3b82f6",
            text=[f"{v:,.0f} kWh" for v in top20["energia_kwh"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>kWh: %{x:,.1f}<extra></extra>",
            name="kWh"
        ),
        row=2, col=1
    )

    # Barras: COP/kWh
    fig.add_trace(
        go.Bar(
            y=top20["evse_uid"],
            x=top20["cop_por_kwh"],
            orientation="h",
            marker_color="#10b981",
            text=[f"${v:,.0f}/kWh" if pd.notna(v) else "—" for v in top20["cop_por_kwh"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>COP/kWh: %{x:,.0f}<extra></extra>",
            name="COP/kWh"
        ),
        row=2, col=2
    )

    # Barras: kWh/trans
    fig.add_trace(
        go.Bar(
            y=top20["evse_uid"],
            x=top20["kwh_por_trans"],
            orientation="h",
            marker_color="#8b5cf6",
            text=[f"{v:,.2f}" if pd.notna(v) else "—" for v in top20["kwh_por_trans"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>kWh/Trans: %{x:,.2f}<extra></extra>",
            name="kWh/Trans"
        ),
        row=3, col=1
    )

    # Barras: kWh/día
    fig.add_trace(
        go.Bar(
            y=top20["evse_uid"],
            x=top20["kwh_por_dia"],
            orientation="h",
            marker_color="#f59e0b",
            text=[f"{v:,.2f}" for v in top20["kwh_por_dia"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>kWh/Día: %{x:,.2f}<extra></extra>",
            name="kWh/Día"
        ),
        row=3, col=2
    )

    # Layout
    fig.update_layout(
        title=dict(
            text="Energía por Estaciones (kWh) y Eficiencia",
            x=0.5, xanchor="center",
            font=dict(size=22, color="#2d3748", family="Arial Black")
        ),
        height=1100,
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=11),
        margin=dict(l=60, r=30, t=80, b=40)
    )
    # Ejes
    for r in (2, 3):
        for c in (1, 2):
            fig.update_xaxes(gridcolor="#e2e8f0", row=r, col=c)
            fig.update_yaxes(gridcolor="#e2e8f0", row=r, col=c, tickfont=dict(size=9))
    fig.update_xaxes(title_text="kWh", row=2, col=1)
    fig.update_xaxes(title_text="COP/kWh", row=2, col=2)
    fig.update_xaxes(title_text="kWh/Trans", row=3, col=1)
    fig.update_xaxes(title_text="kWh/Día", row=3, col=2)

    return fig

# ------------------------------------------------------------------
# Consola: tablas y reconciliaciones
# ------------------------------------------------------------------
def imprimir_tablas_consola(df: pd.DataFrame, agg_est: pd.DataFrame):
    print("\n" + "="*100)
    print("RESUMEN GENERAL DE ENERGÍA")
    print("="*100)

    total_kwh = agg_est["energia_kwh"].sum(skipna=True)
    total_trans = int(agg_est["transacciones"].sum())
    kwh_over_trans = total_kwh / total_trans if total_trans else np.nan
    total_cop = df["ingresos_cop"].sum(skipna=True)
    cop_per_kwh = total_cop / total_kwh if total_kwh else np.nan

    print(f"• Total energía:            {total_kwh:,.1f} kWh")
    print(f"• Total transacciones:      {total_trans:,}")
    print(f"• kWh promedio / trans:     {kwh_over_trans:,.2f} kWh/trans")
    if pd.notna(total_cop):
        print(f"• Ingresos totales (COP):   ${total_cop:,.0f}")
        print(f"• COP por kWh (global):     ${cop_per_kwh:,.0f}/kWh")

    # Top 10 estaciones por kWh
    print("\n" + "-"*100)
    print("TOP 10 ESTACIONES POR ENERGÍA (kWh)")
    print("-"*100)
    print(f"{'Estación':<34} {'kWh':>12} {'Trans':>8} {'kWh/Trans':>12} {'COP/kWh':>12} {'Días':>6}")
    for _, r in agg_est.head(10).iterrows():
        print(f"{str(r['evse_uid'])[:34]:<34} "
              f"{r['energia_kwh']:>12,.0f} "
              f"{int(r['transacciones']):>8} "
              f"{(r['kwh_por_trans'] if pd.notna(r['kwh_por_trans']) else 0):>12,.2f} "
              f"{(r['cop_por_kwh'] if pd.notna(r['cop_por_kwh']) else 0):>12,.0f} "
              f"{int(r['dias_activo']):>6}")

    # Por tipo de conector
    print("\n" + "-"*100)
    print("RESUMEN POR TIPO DE CONECTOR")
    print("-"*100)
    df_tipo = df.copy()
    df_tipo["tipo_conector"] = df_tipo["evse_uid"].apply(clasificar_conector)
    por_tipo = (
        df_tipo.groupby("tipo_conector")
               .agg(
                   estaciones=("evse_uid", "nunique"),
                   transacciones=("id", "count"),
                   energia_kwh=("energy_kwh", "sum"),
                   ingresos=("ingresos_cop", "sum"),
               )
               .reset_index()
               .sort_values("energia_kwh", ascending=False)
    )
    por_tipo["kWh/Trans"] = por_tipo["energia_kwh"] / por_tipo["transacciones"]
    por_tipo["COP/kWh"] = por_tipo.apply(
        lambda r: (r["ingresos"] / r["energia_kwh"]) if r["energia_kwh"] else np.nan, axis=1
    )

    print(f"{'Tipo':<10} {'Estaciones':>11} {'Trans':>10} {'kWh':>14} {'kWh/Trans':>12} {'COP/kWh':>12}")
    for _, r in por_tipo.iterrows():
        print(f"{r['tipo_conector']:<10} "
              f"{int(r['estaciones']):>11} "
              f"{int(r['transacciones']):>10,} "
              f"{r['energia_kwh']:>14,.0f} "
              f"{r['kWh/Trans']:>12,.2f} "
              f"{(r['COP/kWh'] if pd.notna(r['COP/kWh']) else 0):>12,.0f}")

# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "="*100)
    print("GRÁFICO 10: ENERGÍA POR ESTACIONES")
    print("="*100)

    # 1) Cargar y normalizar
    df = cargar_df()
    print(f"✓ Registros: {len(df):,}")
    print(f"✓ kWh: min={df['energy_kwh'].min(skipna=True):,.2f} "
          f"p50={df['energy_kwh'].median(skipna=True):,.2f} "
          f"p90={df['energy_kwh'].quantile(0.90):,.2f} "
          f"max={df['energy_kwh'].max(skipna=True):,.2f}")

    # 2) Agregados por estación + matriz temporal
    agg_est = preparar_metricas_estacion(df)
    matriz = crear_matriz_temporal_kwh(df)

    # 3) Gráfica
    fig = crear_grafico(agg_est, matriz)

    # 4) Tablas en consola
    imprimir_tablas_consola(df, agg_est)

    # 5) Guardar y mostrar
    print("\n" + "="*100)
    print("Guardando figura…")
    guardar_grafico(fig, "grafico_10_energia_estaciones.png")
    print("Abriendo figura…")
    fig.show()
