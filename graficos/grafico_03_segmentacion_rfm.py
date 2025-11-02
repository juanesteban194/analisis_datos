# -*- coding: utf-8 -*-
"""
Gráfico 3: Segmentación RFM (datos normalizados)
- Monetario SOLO en miles de COP (eje y hover)
- Segmentos en español
- Recorte de outliers al p99 para mejorar legibilidad
- Sin búsquedas dinámicas: columnas fijas -> user_id, start_date_time, amount_transaction
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------------------------------------------------
# Configuración de rutas y utilidades (opcional)
# ---------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PARENT_DIR)

CSV_PATH = os.path.join(PARENT_DIR, "data", "df_oasis_ready.csv")  

try:
    from utils import cargar_datos as _cargar_datos, guardar_grafico as _guardar_grafico
except Exception:
    _cargar_datos = None
    _guardar_grafico = None


def cargar_datos_local(csv_path: str) -> pd.DataFrame:
    """Carga simple con columnas fijas."""
    df = pd.read_csv(csv_path, low_memory=False)
    # Columnas fijas
    df["start_date_time"] = pd.to_datetime(df["start_date_time"], errors="coerce")
    df["amount_transaction"] = pd.to_numeric(df["amount_transaction"], errors="coerce")
    if df["amount_transaction"].isna().all():
        raise ValueError("amount_transaction está vacío o no es numérico.")
    if "user_id" not in df.columns:
        raise ValueError("No se encontró la columna 'user_id' en el CSV.")
    return df


def guardar_grafico_local(fig, filename: str, outdir: str = "outputs"):
    """Guarda PNG (si hay kaleido) o HTML como fallback."""
    os.makedirs(outdir, exist_ok=True)
    png_path = os.path.join(outdir, filename)
    html_path = os.path.splitext(png_path)[0] + ".html"
    try:
        fig.write_image(png_path, scale=2)  # requiere 'kaleido'
        print(f"✓ Gráfico guardado como PNG: {png_path}")
    except Exception:
        fig.write_html(html_path)
        print(f"⚠️ No se pudo guardar PNG (falta kaleido). Guardado HTML: {html_path}")


# Selección final de funciones utilitarias
cargar_datos = _cargar_datos if _cargar_datos else cargar_datos_local
guardar_grafico = _guardar_grafico if _guardar_grafico else guardar_grafico_local

# ---------------------------------------------------------------------
# RFM
# ---------------------------------------------------------------------
def calcular_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula R, F, M por usuario con columnas fijas."""
    fecha_max = df["start_date_time"].max()
    rfm_rows = []
    for user_id, g in df.groupby("user_id"):
        recency = (fecha_max - g["start_date_time"].max()).days
        frequency = len(g)
        monetary = g["amount_transaction"].sum()
        rfm_rows.append({"user_id": user_id, "recency": recency, "frequency": frequency, "monetary": monetary})

    rfm = pd.DataFrame(rfm_rows)

    # Scores (1 a 5) con inversión de Recency (menos días = mejor)
    def _qcut_or_cut(s, labels_if_cut, invert=False):
        try:
            res = pd.qcut(s, q=5, labels=False, duplicates="drop") + 1
        except ValueError:
            res = pd.cut(s, bins=5, labels=labels_if_cut, duplicates="drop").astype(int)
        if invert:
            res = 6 - res
        return res

    rfm["r_score"] = _qcut_or_cut(rfm["recency"], [5, 4, 3, 2, 1], invert=True)
    rfm["f_score"] = _qcut_or_cut(rfm["frequency"], [1, 2, 3, 4, 5])
    rfm["m_score"] = _qcut_or_cut(rfm["monetary"], [1, 2, 3, 4, 5])
    rfm["rfm_score"] = rfm["r_score"] + rfm["f_score"] + rfm["m_score"]

    # Segmentos en español
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


# ---------------------------------------------------------------------
# Gráficos (Monetario en miles de COP con tope p99)
# ---------------------------------------------------------------------
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

def crear_grafico_3d(rfm_df: pd.DataFrame) -> go.Figure:
    d = rfm_df.copy()
    d["monetary_k"] = (d["monetary"] / 1_000.0).round(0)  # SOLO miles

    z_p99 = float(np.nanpercentile(d["monetary_k"], 99))
    d.loc[d["monetary_k"] > z_p99, "monetary_k"] = z_p99

    d["recency_text"] = d["recency"].apply(lambda x: f"{int(x)} días")
    d["frequency_text"] = d["frequency"].apply(lambda x: f"{int(x)} transacciones")
    d["monetary_k_txt"] = d["monetary_k"].apply(lambda x: f"{int(x):,} mil COP")

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
                marker=dict(size=6, color=COLOR_MAP_ES.get(seg, "#64748b"),
                            opacity=0.8, line=dict(width=0.5, color="white")),
                customdata=ds[["user_id", "recency_text", "frequency_text", "monetary_k_txt"]].values,
                hovertemplate=(
                    "<b>Segmento: " + seg + "</b><br>"
                    "Usuario: %{customdata[0]}<br>"
                    "Recencia: %{customdata[1]}<br>"
                    "Frecuencia: %{customdata[2]}<br>"
                    "Monetario: %{customdata[3]}<br>"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=dict(text="Segmentación RFM de Usuarios - Vista 3D Interactiva",
                   x=0.5, xanchor="center",
                   font=dict(size=20, color="#2d3748", family="Arial Black")),
        scene=dict(
            xaxis=dict(title="Recencia (días desde última compra)",
                       backgroundcolor="rgb(230,230,230)", gridcolor="white"),
            yaxis=dict(title="Frecuencia (número de transacciones)",
                       backgroundcolor="rgb(230,230,230)", gridcolor="white"),
            zaxis=dict(
                title="Monetario (miles de COP)",
                tickformat=",.0f",
                range=[0, z_p99],
                backgroundcolor="rgb(230,230,230)", gridcolor="white",
            ),
        ),
        height=800, showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                    bgcolor="rgba(255,255,255,0.8)"),
        font=dict(family="Arial", size=12),
    )
    return fig


def crear_grafico_2d(rfm_df: pd.DataFrame) -> px.scatter:
    d = rfm_df.copy()
    d["monetary_k"] = (d["monetary"] / 1_000.0).round(0)
    z_p99 = float(np.nanpercentile(d["monetary_k"], 99))
    d.loc[d["monetary_k"] > z_p99, "monetary_k"] = z_p99

    fig = px.scatter(
        d,
        x="frequency",
        y="monetary_k",
        color="segment",
        size="recency",
        size_max=20,
        color_discrete_map=COLOR_MAP_ES,
        hover_data={
            "user_id": True,
            "recency": True,
            "rfm_score": True,
            "monetary_k": True,   # miles en hover
            "monetary": False     # oculto COP completos
        },
        labels={
            "frequency": "Frecuencia (Transacciones)",
            "monetary_k": "Monetario (miles de COP)",
            "segment": "Segmento"
        },
        title="Segmentación RFM - Frecuencia vs Monetario (tamaño = Recencia)"
    )

    fig.update_layout(
        title=dict(x=0.5, xanchor="center",
                   font=dict(size=18, color="#2d3748", family="Arial Black")),
        plot_bgcolor="white", paper_bgcolor="white", height=700,
        font=dict(family="Arial", size=12),
    )
    fig.update_xaxes(gridcolor="#e2e8f0")
    fig.update_yaxes(gridcolor="#e2e8f0", tickformat=",.0f")
    return fig


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("GRÁFICO 3: SEGMENTACIÓN RFM (DATOS NORMALIZADOS)")
    print("=" * 80)

    # Cargar datos (columnas fijas)
    df = cargar_datos(CSV_PATH)
    print(f"✓ Archivo: {CSV_PATH} | Registros: {len(df):,}")

    # Cálculo RFM
    print("\n📊 Calculando RFM...")
    rfm_df = calcular_rfm(df)
    print(f"✓ RFM para {len(rfm_df):,} usuarios")

    # Gráficos
    print("🎨 Creando gráficos...")
    fig_3d = crear_grafico_3d(rfm_df)
    fig_2d = crear_grafico_2d(rfm_df)

    # Guardado
    print("\n💾 Guardando gráficos...")
    guardar_grafico(fig_3d, "grafico_03_rfm_3d.png")
    guardar_grafico(fig_2d, "grafico_03_rfm_2d.png")

    # Mostrar (opcional)
    print("🌐 Abriendo gráfico 3D…")
    fig_3d.show()
    print("🌐 Abriendo gráfico 2D…")
    fig_2d.show()

    print("\n" + "=" * 80)
    print("✅ GRÁFICO 3 COMPLETADO")
    print("=" * 80 + "\n")
