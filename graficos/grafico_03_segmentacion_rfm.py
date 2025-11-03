# -*- coding: utf-8 -*-
"""
Gráfico 3: Segmentación RFM (COP reales → mostrados en miles)
- Base monetaria PRIORITARIA: amount_transaction_num (COP reales)
- Fallbacks: ingresos_cop → amount_transaction_cop → amount_transaction/100
- Ejes y hover: miles de COP (amount_k)
- Segmentos en español (coherentes con gráfico 4)
- Reconciliación de totales en consola
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# -----------------------------------------------------------------------------
# Paths / utils
# -----------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))   # .../graficos
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)                 # .../analisis_datos
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import cargar_datos, guardar_grafico, ESTILO_GRAFICO  # type: ignore

# archivos candidatos
CSV_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "data", "df_oasis_ready.csv"),
    os.path.join(PROJECT_ROOT, "data", "df_oasis_clean_normalized.csv"),
    os.path.join(PROJECT_ROOT, "data", "df_oasis_clean.csv"),
]

# Colores RFM en español
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


# -----------------------------------------------------------------------------
# Cargar y normalizar datos
# -----------------------------------------------------------------------------
def _pick_existing_path(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("No se encontró ningún CSV en /data.")

def cargar_df() -> pd.DataFrame:
    """
    Carga el CSV disponible y crea SIEMPRE:
      - start_date_time (datetime)
      - ingresos_cop  (COP reales)
      - amount_k      (miles de COP) -> solo para mostrar
    """
    csv_path = _pick_existing_path(CSV_CANDIDATES)
    df = cargar_datos(csv_path)
    print(f"✓ Usando archivo: {csv_path}")

    # columnas mínimas
    if "user_id" not in df.columns:
        raise ValueError("Falta la columna 'user_id'.")
    if "start_date_time" not in df.columns:
        raise ValueError("Falta la columna 'start_date_time'.")

    df["start_date_time"] = pd.to_datetime(df["start_date_time"], errors="coerce")
    df = df.dropna(subset=["start_date_time"])

    # ====== MONETARIO ======
    # prioridad 1: amount_transaction_num (la que ya validamos que es la correcta)
    if "amount_transaction_num" in df.columns:
        df["ingresos_cop"] = pd.to_numeric(df["amount_transaction_num"], errors="coerce").fillna(0)
    # prioridad 2
    elif "ingresos_cop" in df.columns:
        df["ingresos_cop"] = pd.to_numeric(df["ingresos_cop"], errors="coerce").fillna(0)
    # prioridad 3
    elif "amount_transaction_cop" in df.columns:
        df["ingresos_cop"] = pd.to_numeric(df["amount_transaction_cop"], errors="coerce").fillna(0)
    # último recurso: viene en centavos → /100
    elif "amount_transaction" in df.columns:
        df["ingresos_cop"] = pd.to_numeric(df["amount_transaction"], errors="coerce").fillna(0) / 100.0
    else:
        raise ValueError(
            "No encontré ninguna columna monetaria. Necesito "
            "'amount_transaction_num', 'ingresos_cop', 'amount_transaction_cop' o 'amount_transaction'."
        )

    # para mostrar: miles
    df["amount_k"] = df["ingresos_cop"] / 1_000.0

    # pequeño chequeo
    print(
        "✓ Chequeo montos (miles):",
        "min", round(df["amount_k"].min(), 2),
        "p50", round(df["amount_k"].median(), 2),
        "p90", round(df["amount_k"].quantile(0.90), 2),
        "max", round(df["amount_k"].max(), 2),
    )

    return df


# -----------------------------------------------------------------------------
# Cálculo RFM (monetario en miles SOLO para graficar)
# -----------------------------------------------------------------------------
def calcular_rfm(df: pd.DataFrame) -> pd.DataFrame:
    fecha_max = df["start_date_time"].max()

    rows = []
    for user_id, g in df.groupby("user_id"):
        recency = (fecha_max - g["start_date_time"].max()).days
        frequency = len(g)
        # miles de COP (solo para eje); el total REAL lo reconciliaremos con df["ingresos_cop"]
        monetary_k = g["amount_k"].sum()
        rows.append(
            {
                "user_id": user_id,
                "recency": recency,
                "frequency": frequency,
                "monetary_k": monetary_k,
            }
        )

    rfm = pd.DataFrame(rows)

    # ---- scores ----
    def _qcut_or_cut(s, labels_if_cut, invert=False):
        s = pd.to_numeric(s, errors="coerce").fillna(0)
        try:
            res = pd.qcut(s, q=5, labels=False, duplicates="drop") + 1
        except ValueError:
            res = pd.cut(s, bins=5, labels=labels_if_cut, duplicates="drop").astype(int)
        if invert:
            res = 6 - res
        return res

    rfm["r_score"] = _qcut_or_cut(rfm["recency"], [5, 4, 3, 2, 1], invert=True)
    rfm["f_score"] = _qcut_or_cut(rfm["frequency"], [1, 2, 3, 4, 5])
    rfm["m_score"] = _qcut_or_cut(rfm["monetary_k"], [1, 2, 3, 4, 5])
    rfm["rfm_score"] = rfm["r_score"] + rfm["f_score"] + rfm["m_score"]

    # ---- segmentación coherente con gráfico 4 ----
    def segmentar(row):
        r, f, m, s = row["r_score"], row["f_score"], row["m_score"], row["rfm_score"]
        if s >= 13:
            return "Campeones"
        elif f >= 4:
            return "Leales"
        elif r >= 4 and f >= 3:
            return "Leal potencial"
        elif r <= 2 and f >= 3:
            return "En riesgo"
        elif r <= 2 and m >= 4:
            return "No podemos perder"
        elif r <= 2:
            return "Hibernando"
        elif r >= 4 and f <= 2:
            return "Prometedor"
        else:
            return "Necesita atención"

    rfm["segment"] = rfm.apply(segmentar, axis=1)
    return rfm


# -----------------------------------------------------------------------------
# Gráfico 3D
# -----------------------------------------------------------------------------
def crear_grafico_3d(rfm_df: pd.DataFrame) -> go.Figure:
    d = rfm_df.copy()

    # recorte visual
    z_p99 = float(np.nanpercentile(d["monetary_k"], 99))
    d.loc[d["monetary_k"] > z_p99, "monetary_k"] = z_p99

    fig = go.Figure()
    for seg in sorted(d["segment"].unique()):
        ds = d[d["segment"] == seg]
        fig.add_trace(
            go.Scatter3d(
                x=ds["recency"],
                y=ds["frequency"],
                z=ds["monetary_k"],
                mode="markers",
                name=seg,
                marker=dict(
                    size=6,
                    color=COLOR_MAP_ES.get(seg, "#64748b"),
                    opacity=0.85,
                    line=dict(width=0.5, color="white"),
                ),
                customdata=ds[["user_id", "recency", "frequency", "monetary_k"]].values,
                hovertemplate=(
                    "<b>Segmento: " + seg + "</b><br>"
                    "Usuario: %{customdata[0]}<br>"
                    "Recencia: %{customdata[1]} días<br>"
                    "Frecuencia: %{customdata[2]}<br>"
                    "Monetario: %{customdata[3]:,.2f} mil COP<br><extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=dict(
            text="Segmentación RFM de Usuarios - Vista 3D",
            x=0.5, xanchor="center",
            font=dict(
                family=ESTILO_GRAFICO.get("font_family", "Arial"),
                size=20,
                color="#2d3748",
            ),
        ),
        scene=dict(
            xaxis=dict(title="Recencia (días)"),
            yaxis=dict(title="Frecuencia (transacciones)"),
            zaxis=dict(title="Monetario (miles de COP)", tickformat=",.0f"),
        ),
        height=820,
        showlegend=True,
        paper_bgcolor=ESTILO_GRAFICO.get("bg_color", "white"),
        plot_bgcolor=ESTILO_GRAFICO.get("bg_color", "white"),
    )
    return fig


# -----------------------------------------------------------------------------
# Gráfico 2D
# -----------------------------------------------------------------------------
def crear_grafico_2d(rfm_df: pd.DataFrame) -> px.scatter:
    d = rfm_df.copy()
    z_p99 = float(np.nanpercentile(d["monetary_k"], 99))
    d.loc[d["monetary_k"] > z_p99, "monetary_k"] = z_p99

    fig = px.scatter(
        d,
        x="frequency",
        y="monetary_k",
        color="segment",
        size="recency",
        size_max=22,
        color_discrete_map=COLOR_MAP_ES,
        hover_data=["user_id", "recency", "rfm_score", "monetary_k"],
        labels={
            "frequency": "Frecuencia (transacciones)",
            "monetary_k": "Monetario (miles de COP)",
            "segment": "Segmento",
        },
        title="Segmentación RFM - Frequency vs Monetary",
    )

    fig.update_layout(
        xaxis=dict(gridcolor="#e2e8f0"),
        yaxis=dict(gridcolor="#e2e8f0", tickformat=",.0f"),
        height=720,
        paper_bgcolor=ESTILO_GRAFICO.get("bg_color", "white"),
        plot_bgcolor=ESTILO_GRAFICO.get("bg_color", "white"),
    )
    return fig


# -----------------------------------------------------------------------------
# Consola
# -----------------------------------------------------------------------------
def imprimir_resumen_consola(rfm_df: pd.DataFrame, df_base: pd.DataFrame) -> None:
    seg_stats = (
        rfm_df.groupby("segment")
        .agg(
            usuarios=("user_id", "count"),
            recency_avg=("recency", "mean"),
            frequency_avg=("frequency", "mean"),
            monetary_avg_k=("monetary_k", "mean"),
            rfm_score_avg=("rfm_score", "mean"),
        )
        .round(2)
        .sort_values("usuarios", ascending=False)
    )
    total_usuarios = len(rfm_df)

    print("\n" + "=" * 90)
    print("📊 RESUMEN POR SEGMENTO (monetario en MILES de COP)")
    print("=" * 90)
    print(f"{'Segmento':<20} {'Usuarios':>8} {'%':>6} {'Rec':>6} {'Freq':>6} {'$ promedio (mil)':>18}")
    print("-" * 90)
    for seg, r in seg_stats.iterrows():
        pct = r["usuarios"] / total_usuarios * 100
        print(
            f"{seg:<20} {int(r['usuarios']):>8} {pct:>5.1f} "
            f"{r['recency_avg']:>6.1f} {r['frequency_avg']:>6.1f} {r['monetary_avg_k']:>18,.2f}"
        )

    # ---- Reconciliación real en COP ----
    total_cop = df_base["ingresos_cop"].sum(skipna=True)
    print("\n" + "=" * 90)
    print(f"🧮 Total de ingresos en dataset (COP reales): {total_cop:,.0f}")
    print("    (esto es el valor que debe concordar con tus ~166 millones)")
    print("=" * 90)


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 110)
    print("GRÁFICO 3: SEGMENTACIÓN RFM (base en COP reales, ejes en MILES)")
    print("=" * 110)

    df = cargar_df()
    rfm_df = calcular_rfm(df)

    print("🎨 Creando gráficos…")
    fig_3d = crear_grafico_3d(rfm_df)
    fig_2d = crear_grafico_2d(rfm_df)

    print("💾 Guardando gráficos…")
    guardar_grafico(fig_3d, "grafico_03_rfm_3d.png")
    guardar_grafico(fig_2d, "grafico_03_rfm_2d.png")

    print("🌐 Mostrando…")
    fig_3d.show()
    fig_2d.show()

    imprimir_resumen_consola(rfm_df, df)

    print("\n✅ Gráfico 3 listo y coherente con el total real.\n")
