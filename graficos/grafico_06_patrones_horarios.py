# -*- coding: utf-8 -*-
"""
Gráfico 6: Patrones Horarios (heatmap 7×24 consistente)
- Cuenta de transacciones por hora (0–23) y día (Lun–Dom)
- Malla completa con ceros donde no hubo actividad
- Títulos y etiquetas en español
"""

import os, sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- rutas / utils ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import cargar_datos, guardar_grafico  # noqa

CSV_CANDIDATOS = [
    os.path.join(PROJECT_ROOT, "data", "df_oasis_ready.csv"),
    os.path.join(PROJECT_ROOT, "data", "df_oasis_clean_normalized.csv"),
    os.path.join(PROJECT_ROOT, "data", "df_oasis_clean.csv"),
]

DIAS_ES = ["Lunes","Martes","Miércoles","Jueves","Viernes","Sábado","Domingo"]
MAP_DIAS_ES = {
    "Monday":"Lunes","Tuesday":"Martes","Wednesday":"Miércoles",
    "Thursday":"Jueves","Friday":"Viernes","Saturday":"Sábado","Sunday":"Domingo"
}

# ---------------- carga / preparación ----------------
def _elegir_csv():
    for p in CSV_CANDIDATOS:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("No se encontró CSV en /data.")

def preparar_datos_temporales(df: pd.DataFrame) -> pd.DataFrame:
    if "start_date_time" not in df.columns:
        raise ValueError("Falta 'start_date_time' en el dataset.")

    d = df.copy()
    d["start_date_time"] = pd.to_datetime(d["start_date_time"], errors="coerce")
    d = d.dropna(subset=["start_date_time"])

    # hora (0–23), día de semana en español y flag fin de semana
    d["hora"] = d["start_date_time"].dt.hour.astype(int)
    d["dia_nombre"] = d["start_date_time"].dt.day_name()
    d["dia_nombre_es"] = d["dia_nombre"].map(MAP_DIAS_ES).fillna(d["dia_nombre"])
    d["es_fin_semana"] = d["dia_nombre_es"].isin(["Sábado","Domingo"])

    # normaliza valores inesperados
    d["hora"] = d["hora"].clip(lower=0, upper=23)
    return d

def crear_matriz_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve matriz 24×7 (index=hora 0–23, columnas=DIAS_ES) con ceros donde no hay datos.
    """
    # conteo por (hora, día)
    base = (df.groupby(["hora","dia_nombre_es"]).size()
              .rename("transacciones")
              .reset_index())

    # malla completa 24×7
    malla = pd.MultiIndex.from_product([range(24), DIAS_ES], names=["hora","dia_nombre_es"])
    base = base.set_index(["hora","dia_nombre_es"]).reindex(malla, fill_value=0).reset_index()

    # pivot a matriz
    matriz = (base.pivot(index="hora", columns="dia_nombre_es", values="transacciones")
                   .reindex(columns=DIAS_ES, fill_value=0)
                   .sort_index())
    return matriz

# ---------------- gráficos ----------------
def crear_grafico(df: pd.DataFrame, matriz: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Heatmap de Uso: Hora × Día de la Semana",
            "Transacciones por Hora del Día",
            "Transacciones por Día de la Semana",
            "Comparación: Días Laborales vs Fin de Semana"
        ),
        specs=[[{"type":"heatmap","colspan":2}, None],
               [{"type":"bar"}, {"type":"bar"}]],
        row_heights=[0.6, 0.4], vertical_spacing=0.15, horizontal_spacing=0.15
    )

    # Heatmap principal (24×7 garantizado)
    fig.add_trace(
        go.Heatmap(
            z=matriz.values,
            x=matriz.columns.tolist(),
            y=matriz.index.tolist(),
            colorscale="Blues",
            hovertemplate="<b>%{x}</b><br>Hora: %{y}:00<br>Transacciones: %{z}<extra></extra>",
            colorbar=dict(title="Transacciones", x=1.02),
            zmin=0, zmax=max(1, int(matriz.values.max()))  # escala estable
        ),
        row=1, col=1
    )

    # Transacciones por hora (suma de columnas)
    trans_por_hora = matriz.sum(axis=1)
    fig.add_trace(
        go.Bar(
            x=trans_por_hora.index, y=trans_por_hora.values,
            marker_color="#3b82f6", name="Por hora",
            hovertemplate="<b>%{x}:00</b><br>Transacciones: %{y}<extra></extra>"
        ),
        row=2, col=1
    )

    # Transacciones por día (suma de filas)
    trans_por_dia = matriz.sum(axis=0).reindex(DIAS_ES)
    colors_dias = ["#3b82f6"]*5 + ["#10b981"]*2  # laborales vs finde
    fig.add_trace(
        go.Bar(
            x=trans_por_dia.index, y=trans_por_dia.values,
            marker_color=colors_dias, name="Por día",
            hovertemplate="<b>%{x}</b><br>Transacciones: %{y}<extra></extra>"
        ),
        row=2, col=2
    )

    fig.update_layout(
        title=dict(text="Análisis de Patrones Horarios de Uso",
                   x=0.5, xanchor="center",
                   font=dict(size=20, color="#2d3748", family="Arial Black")),
        showlegend=False, height=1000,
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Arial", size=11)
    )
    fig.update_xaxes(title_text="Día de la Semana", row=1, col=1)
    fig.update_yaxes(title_text="Hora del Día", row=1, col=1)

    fig.update_xaxes(title_text="Hora", row=2, col=1)
    fig.update_yaxes(title_text="Transacciones", gridcolor="#e2e8f0", row=2, col=1)

    fig.update_xaxes(title_text="Día", tickangle=-45, row=2, col=2)
    fig.update_yaxes(title_text="Transacciones", gridcolor="#e2e8f0", row=2, col=2)
    return fig

def crear_grafico_horario_detallado(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for dia in DIAS_ES:
        # serie 0–23 sin huecos
        serie = (df.loc[df["dia_nombre_es"]==dia]
                   .groupby("hora").size().reindex(range(24), fill_value=0))
        color = "#10b981" if dia in ["Sábado","Domingo"] else "#3b82f6"
        width = 3 if dia in ["Sábado","Domingo"] else 2
        fig.add_trace(go.Scatter(
            x=serie.index, y=serie.values, mode="lines+markers",
            name=dia, line=dict(color=color, width=width), marker=dict(size=6)
        ))
    fig.update_layout(
        title=dict(text="Patrones Horarios por Día de la Semana",
                   x=0.5, xanchor="center",
                   font=dict(size=18, color="#2d3748", family="Arial Black")),
        xaxis_title="Hora del Día", yaxis_title="Número de Transacciones",
        plot_bgcolor="white", paper_bgcolor="white",
        height=600, font=dict(family="Arial", size=12), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(gridcolor="#e2e8f0", range=[0,23])
    fig.update_yaxes(gridcolor="#e2e8f0")
    return fig

# ---------------- análisis ----------------
def analizar(df: pd.DataFrame, matriz: pd.DataFrame) -> dict:
    total = int(len(df))

    trans_por_hora = matriz.sum(axis=1)
    hora_pico = int(trans_por_hora.idxmax())
    hora_valle = int(trans_por_hora.idxmin())

    trans_hora_pico = int(trans_por_hora.max())
    trans_hora_valle = int(trans_por_hora.min())

    trans_por_dia = matriz.sum(axis=0).reindex(DIAS_ES)
    dia_pico = str(trans_por_dia.idxmax())
    trans_dia_pico = int(trans_por_dia.max())

    trans_laborales = int(trans_por_dia.iloc[:5].sum())
    trans_fin_semana = int(trans_por_dia.iloc[5:].sum())
    pct_laborales = (trans_laborales/total*100) if total else 0
    pct_fin_semana = (trans_fin_semana/total*100) if total else 0

    promedio_dia_laboral = trans_laborales/5
    promedio_dia_fin_semana = trans_fin_semana/2

    # Franjas
    rangos = {
        "mañana":   range(6,12),
        "mediodía": range(12,15),
        "tarde":    range(15,20),
        "noche":    range(20,24),
        "madrugada":range(0,6),
    }
    conta = {k: int(trans_por_hora.loc[list(v)].sum()) for k,v in rangos.items()}
    pct   = {k: (conta[k]/total*100 if total else 0) for k in conta}
    franja_top = max(conta, key=conta.get)

    # Punto más caliente (hora, día) en la matriz
    stacked = matriz.stack()
    (p_hora, p_dia) = stacked.idxmax()
    p_val = int(stacked.max())

    return {
        "total_transacciones": total,
        "hora_pico": hora_pico, "trans_hora_pico": trans_hora_pico, "pct_hora_pico": (trans_hora_pico/total*100 if total else 0),
        "hora_valle": hora_valle, "trans_hora_valle": trans_hora_valle,
        "dia_pico": dia_pico, "trans_dia_pico": trans_dia_pico,
        "trans_laborales": trans_laborales, "pct_laborales": pct_laborales, "promedio_dia_laboral": promedio_dia_laboral,
        "trans_fin_semana": trans_fin_semana, "pct_fin_semana": pct_fin_semana, "promedio_dia_fin_semana": promedio_dia_fin_semana,
        "trans_manana": conta["mañana"], "pct_manana": pct["mañana"],
        "trans_mediodia": conta["mediodía"], "pct_mediodia": pct["mediodía"],
        "trans_tarde": conta["tarde"], "pct_tarde": pct["tarde"],
        "trans_noche": conta["noche"], "pct_noche": pct["noche"],
        "trans_madrugada": conta["madrugada"], "pct_madrugada": pct["madrugada"],
        "punto_max_hora": int(p_hora), "punto_max_dia": str(p_dia), "punto_max_valor": p_val,
        "insight_hora_pico": f"La hora pico es {hora_pico}:00 con {trans_hora_pico} transacciones ({(trans_hora_pico/total*100 if total else 0):.1f}% del total).",
        "insight_dia_pico": f"El día más activo es {dia_pico} con {trans_dia_pico} transacciones.",
        "insight_laboral_vs_finde": f"Los días laborales concentran {pct_laborales:.1f}% vs {pct_fin_semana:.1f}% en fin de semana. Promedio: {promedio_dia_laboral:.0f} (laboral) vs {promedio_dia_fin_semana:.0f} (finde).",
        "insight_franjas": f"La franja más activa es la {franja_top} ({conta[franja_top]} transacciones).",
        "insight_punto_caliente": f"El punto más activo es {p_dia} a las {p_hora}:00 con {p_val} transacciones."
    }

# ---------------- MAIN ----------------
if __name__ == "__main__":
    print("\n" + "="*80)
    print("GRÁFICO 6: PATRONES HORARIOS")
    print("="*80)

    csv_path = _elegir_csv()
    df = cargar_datos(csv_path)
    print(f"✓ Datos cargados: {len(df):,} registros desde {os.path.basename(csv_path)}\n")

    print("Preparando datos temporales…")
    df_t = preparar_datos_temporales(df)
    print("✓ Datos temporales listos\n")

    print("Construyendo matriz 24×7…")
    matriz = crear_matriz_heatmap(df_t)
    print(f"✓ Matriz lista (suma matriz = {int(matriz.values.sum())} trans)")

    # Verificación de consistencia con el total del df
    total_df = len(df_t)
    total_mat = int(matriz.values.sum())
    print(f"✓ Chequeo totales: df={total_df} | heatmap={total_mat} | diff={total_df - total_mat}")

    print("Creando gráficos…")
    fig_main = crear_grafico(df_t, matriz)
    fig_detail = crear_grafico_horario_detallado(df_t)

    print("\n" + "="*80)
    print("ANÁLISIS DE PATRONES HORARIOS")
    print("="*80)
    insights = analizar(df_t, matriz)

    # Resumen en consola
    print(f"\nTOTAL TRANSACCIONES: {insights['total_transacciones']:,}")
    print(f"• Hora pico: {insights['hora_pico']}:00 — {insights['trans_hora_pico']} ({insights['pct_hora_pico']:.1f}%)")
    print(f"• Hora valle: {insights['hora_valle']}:00 — {insights['trans_hora_valle']}")
    print(f"• Día más activo: {insights['dia_pico']} — {insights['trans_dia_pico']}")
    print(f"• Laborales: {insights['trans_laborales']:,} ({insights['pct_laborales']:.1f}%)  |  Finde: {insights['trans_fin_semana']:,} ({insights['pct_fin_semana']:.1f}%)")
    print(f"• Promedios: laboral {insights['promedio_dia_laboral']:.0f}  |  finde {insights['promedio_dia_fin_semana']:.0f}")
    print(f"• Franjas → mañana {insights['trans_manana']}, mediodía {insights['trans_mediodia']}, tarde {insights['trans_tarde']}, noche {insights['trans_noche']}, madrugada {insights['trans_madrugada']}")
    print(f"• Punto caliente: {insights['punto_max_dia']} a las {insights['punto_max_hora']}:00 — {insights['punto_max_valor']}")

    print("\nGuardando gráficos…")
    guardar_grafico(fig_main, "grafico_06_patrones_horarios.png")
    guardar_grafico(fig_detail, "grafico_06_horario_detallado.png")

    print("Abriendo gráficos…")
    fig_main.show()
    fig_detail.show()
