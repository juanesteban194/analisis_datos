# -*- coding: utf-8 -*-
"""
Gráfico 4: CLV por Segmentos (COP reales)
- Fuente monetaria: amount_transaction_num (COP reales, sin dividir por 100)
- Segmentación RFM en español
- CLV proyectado a 1 año por usuario y promedio por segmento
- Barras comparativas: (1) CLV promedio (MM), (2) Ingresos totales (MM),
  (3) Número de usuarios, (4) Valor promedio por transacción (MM)
- Reconciliación de totales en consola (segmentos vs total global)
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------- Paths / utils del proyecto ----------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from utils import cargar_datos, guardar_grafico 

# ---------------- Config ----------------
CSV_CANDIDATOS = [
    os.path.join(PARENT_DIR, "data", "df_oasis_ready.csv"),
    os.path.join(PARENT_DIR, "data", "df_oasis_clean_normalized.csv"),
    os.path.join(PARENT_DIR, "data", "df_oasis_clean.csv"),
]

COLOR_MAP_ES = {
    "Campeones": "#10b981",
    "Leales": "#3b82f6",
    "Leal potencial": "#8b5cf6",
    "En riesgo": "#ef4444",
    "No podemos perder": "#dc2626",
    "Hibernando": "#6b7280",
    "Prometedor": "#f59e0b",
    "Necesita atención": "#f97316",
}

# ---------------- Helpers ----------------
def _elegir_csv() -> str:
    for p in CSV_CANDIDATOS:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("No se encontró ningún CSV en /data.")

def _qcut_or_cut(s: pd.Series, labels_if_cut, invert=False) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0)
    try:
        res = pd.qcut(s, q=5, labels=False, duplicates="drop") + 1
    except ValueError:
        res = pd.cut(s, bins=5, labels=labels_if_cut, duplicates="drop").astype(int)
    if invert:
        res = 6 - res
    return res

# ---------------- Preparación de datos ----------------
def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:
    # Fechas
    for col in ("start_date_time", "end_date_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Asegurar columnas clave
    if "user_id" not in df.columns:
        raise ValueError("Falta 'user_id' en el dataset.")
    if "amount_transaction_num" not in df.columns:
        raise ValueError("Falta 'amount_transaction_num' (COP reales).")

    # Monetario en COP (reales)
    df["monto_cop"] = pd.to_numeric(df["amount_transaction_num"], errors="coerce").fillna(0)

    # Conversión a escalas para métricas/plots
    df["monto_miles"] = df["monto_cop"] / 1_000.0        # miles de COP
    df["monto_MM"]    = df["monto_cop"] / 1_000_000.0    # millones de COP

    return df

# ---------------- Segmentación RFM ----------------
def calcular_rfm(df: pd.DataFrame) -> pd.DataFrame:
    if "start_date_time" not in df.columns:
        raise ValueError("Se requiere 'start_date_time' para Recency.")

    fecha_max = df["start_date_time"].max()
    rows = []
    for user_id, g in df.groupby("user_id"):
        rec = int((fecha_max - g["start_date_time"].max()).days)
        freq = int(len(g))
        mon_miles = float(g["monto_miles"].sum())  # miles de COP
        rows.append({"user_id": user_id, "recency": rec, "frequency": freq, "monetary_miles": mon_miles})

    rfm = pd.DataFrame(rows)
    rfm["r_score"] = _qcut_or_cut(rfm["recency"], [5, 4, 3, 2, 1], invert=True)   # menor recency = mejor
    rfm["f_score"] = _qcut_or_cut(rfm["frequency"], [1, 2, 3, 4, 5])
    rfm["m_score"] = _qcut_or_cut(rfm["monetary_miles"], [1, 2, 3, 4, 5])
    rfm["rfm_score"] = rfm["r_score"] + rfm["f_score"] + rfm["m_score"]

    def segmentar(row):
        r, f, m, s = row["r_score"], row["f_score"], row["m_score"], row["rfm_score"]
        if s >= 13: return "Campeones"
        elif f >= 4: return "Leales"
        elif r >= 4 and f >= 3: return "Leal potencial"
        elif r <= 2 and f >= 3: return "En riesgo"
        elif r <= 2 and m >= 4: return "No podemos perder"
        elif r <= 2: return "Hibernando"
        elif r >= 4 and f <= 2: return "Prometedor"
        else: return "Necesita atención"

    rfm["segment"] = rfm.apply(segmentar, axis=1)
    return rfm

# ---------------- CLV + agregación por segmento ----------------
def calcular_clv_y_segmentos(df: pd.DataFrame, rfm_df: pd.DataFrame) -> pd.DataFrame:
    # Vida y CLV por usuario (medidas en COP y conversiones)
    life = (
        df.groupby("user_id")
          .agg(
              first=("start_date_time", "min"),
              last=("start_date_time", "max"),
              transacciones=("user_id", "count"),
              gasto_total_cop=("monto_cop", "sum"),
          )
          .reset_index()
    )
    life["lifetime_days"] = (life["last"] - life["first"]).dt.days + 1
    life["lifetime_days"] = life["lifetime_days"].clip(lower=1)

    life["transacciones"] = life["transacciones"].replace(0, np.nan)
    life["avg_tx_value_cop"] = (life["gasto_total_cop"] / life["transacciones"]).fillna(0)
    life["tx_per_year"] = (life["transacciones"] / life["lifetime_days"] * 365.0).replace([np.inf, -np.inf], 0).fillna(0)

    # CLV 1 año por usuario (en COP); para gráfica convertimos a MM
    life["clv_1y_user_cop"] = life["avg_tx_value_cop"] * life["tx_per_year"]
    life["avg_tx_value_MM"]  = life["avg_tx_value_cop"] / 1_000_000.0
    life["clv_1y_user_MM"]   = life["clv_1y_user_cop"] / 1_000_000.0

    users = rfm_df.merge(
        life[["user_id", "lifetime_days", "avg_tx_value_MM", "tx_per_year", "clv_1y_user_MM"]],
        on="user_id", how="left"
    )

    # Ingresos reales por segmento desde transacciones (en COP y MM)
    tx_seg = (
        df[["user_id", "monto_cop"]]
        .merge(rfm_df[["user_id", "segment"]], on="user_id", how="left")
        .groupby("segment", as_index=False)["monto_cop"].sum()
        .rename(columns={"monto_cop": "ingresos_cop"})
    )
    tx_seg["ingresos_MM"] = tx_seg["ingresos_cop"] / 1_000_000.0

    # Promedios por segmento
    seg = (
        users.groupby("segment")
             .agg(
                 num_users=("user_id", "count"),
                 avg_revenue_per_user_miles=("monetary_miles", "mean"),  # miles (info)
                 avg_frequency=("frequency", "mean"),
                 avg_lifetime_days=("lifetime_days", "mean"),
                 avg_transaction_value_MM=("avg_tx_value_MM", "mean"),
                 clv_projected_1year_MM=("clv_1y_user_MM", "mean"),
             )
             .reset_index()
    )

    # Mezclar ingresos reales por segmento (en millones)
    seg = seg.merge(tx_seg, on="segment", how="left").fillna({"ingresos_cop": 0, "ingresos_MM": 0})

    # % por usuarios e ingresos
    total_users = seg["num_users"].sum()
    total_ingresos_cop = seg["ingresos_cop"].sum()
    seg["user_percentage"] = np.where(total_users > 0, seg["num_users"] / total_users * 100, 0.0)
    seg["revenue_share_pct"] = np.where(total_ingresos_cop > 0, seg["ingresos_cop"] / total_ingresos_cop * 100, 0.0)

    # Orden por CLV promedio
    seg = seg.sort_values("clv_projected_1year_MM", ascending=False)

    # Guardar totales en atributos
    seg.attrs["total_global_cop"] = float(total_ingresos_cop)
    seg.attrs["total_global_MM"]  = float(total_ingresos_cop) / 1_000_000.0
    return seg

# ---------------- Gráfico ----------------
def crear_grafico(clv_df: pd.DataFrame) -> go.Figure:
    colors = [COLOR_MAP_ES.get(seg, "#64748b") for seg in clv_df["segment"]]
    total_MM = clv_df.attrs.get("total_global_MM", None)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "CLV promedio por usuario (proyección 1 año, MILLONES COP)",
            "Ingresos totales por segmento (MILLONES COP)",
            "Número de usuarios por segmento",
            "Valor promedio por transacción (MILLONES COP)",
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )

    # 1) CLV promedio (MM)
    fig.add_bar(
        x=clv_df["segment"], y=clv_df["clv_projected_1year_MM"],
        marker_color=colors,
        text=[f"{v:,.2f} M" for v in clv_df["clv_projected_1year_MM"]],
        textposition="outside", name="CLV",
        hovertemplate="<b>%{x}</b><br>CLV prom.: %{y:,.3f} M COP<extra></extra>",
        row=1, col=1
    )

    # 2) Ingresos totales por segmento (MM) – reales del dataset
    fig.add_bar(
        x=clv_df["segment"], y=clv_df["ingresos_MM"],
        marker_color=colors,
        text=[f"{v:,.2f} M" for v in clv_df["ingresos_MM"]],
        textposition="outside", name="Ingresos",
        hovertemplate="<b>%{x}</b><br>Ingresos: %{y:,.3f} M COP<br>Share: %{customdata:.2f}%<extra></extra>",
        customdata=clv_df["revenue_share_pct"].round(2),
        row=1, col=2
    )

    # 3) Usuarios
    fig.add_bar(
        x=clv_df["segment"], y=clv_df["num_users"],
        marker_color=colors,
        text=[f"{n} ({p:.1f}%)" for n, p in zip(clv_df["num_users"], clv_df["user_percentage"])],
        textposition="outside", name="Usuarios",
        hovertemplate="<b>%{x}</b><br>Usuarios: %{y}<extra></extra>",
        row=2, col=1
    )

    # 4) Valor promedio por transacción (MM)
    fig.add_bar(
        x=clv_df["segment"], y=clv_df["avg_transaction_value_MM"],
        marker_color=colors,
        text=[f"{v:,.3f} M" for v in clv_df["avg_transaction_value_MM"]],
        textposition="outside", name="Valor/Trans",
        hovertemplate="<b>%{x}</b><br>Valor prom.: %{y:,.4f} M COP<extra></extra>",
        row=2, col=2
    )

    fig.update_layout(
        title=dict(
            text=("Análisis de Customer Lifetime Value (CLV) por Segmento"
                  + (f" — Total global: {total_MM:,.2f} M COP" if total_MM is not None else "")),
            x=0.5, xanchor="center",
            font=dict(size=20, color="#2d3748", family="Arial Black")
        ),
        showlegend=False, height=900,
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Arial", size=11),
        margin=dict(t=70, r=50, b=50, l=60)
    )
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=10))
    fig.update_yaxes(gridcolor="#e2e8f0")
    return fig

# ---------------- Reconciliación ----------------
def reconciliar_totales(df: pd.DataFrame, clv_df: pd.DataFrame) -> None:
    total_global_MM = clv_df.attrs["total_global_MM"]
    suma_segmentos_MM = float(clv_df["ingresos_MM"].sum())
    diff_MM = total_global_MM - suma_segmentos_MM

    print("\n" + "=" * 100)
    print("RECONCILIACIÓN DE TOTALES (ingresos por segmento vs total global)")
    print("=" * 100)
    print(f"Total global (dataset)             : {total_global_MM:>12,.3f} M COP")
    print(f"Suma segmentos (desde transacciones): {suma_segmentos_MM:>12,.3f} M COP")
    print(f"Diferencia                         : {diff_MM:>12,.6f} M COP "
          f"{'(OK)' if abs(diff_MM) < 1e-6 else '(revisar redondeos)'}")

    print("\nSegmentos (millones y % de participación):")
    print("-" * 100)
    tabla = clv_df[["segment", "ingresos_MM", "revenue_share_pct", "num_users", "user_percentage"]].copy()
    for _, r in tabla.sort_values("ingresos_MM", ascending=False).iterrows():
        print(f"{r['segment']:<20} | {r['ingresos_MM']:>10,.3f} M | share: {r['revenue_share_pct']:>6.2f}% "
              f"| usuarios: {int(r['num_users']):>5} ({r['user_percentage']:>5.1f}%)")

# ---------------- MAIN ----------------
if __name__ == "__main__":
    print("\n" + "="*80)
    print("GRÁFICO 4: CLV POR SEGMENTOS (COP reales)")
    print("="*80)

    csv_path = _elegir_csv()
    df_raw = cargar_datos(csv_path)
    df = preparar_datos(df_raw)

    # RFM + CLV por segmento
    rfm_df = calcular_rfm(df)
    clv_df = calcular_clv_y_segmentos(df, rfm_df)

    # Gráfico
    fig = crear_grafico(clv_df)
    guardar_grafico(fig, "grafico_04_clv_segmentos.png")
    fig.show()

    # Reconciliación
    reconciliar_totales(df, clv_df)
