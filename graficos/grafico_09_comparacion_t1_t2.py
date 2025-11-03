# -*- coding: utf-8 -*-
"""
Gráfico 9: Comparación T1 vs T2
Análisis: Comparación detallada entre tipos de conectores (Tipo 1 vs Tipo 2)
Tipo: Dashboard comparativo multi-métrica

Normalización:
- Ingresos en COP (columna garantizada 'ingresos_cop').
- Si no existe, usa 'amount_transaction_cop'; de no existir, usa 'amount_transaction'/100 (centavos→COP).
- Energía en kWh (columna 'energy_kwh' si existe).
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -------------------------------------------------------------
# Import utils desde la raíz del proyecto (ANALISIS_DATOS/)
# -------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # .../graficos
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)               # .../ANALISIS_DATOS
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import cargar_datos, guardar_grafico  # noqa: E402

# -------------------------------------------------------------
# CARGA ROBUSTA + NORMALIZACIÓN
# -------------------------------------------------------------
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
    raise FileNotFoundError("No se encontró CSV. Probé:\n- " + "\n- ".join(paths))

def cargar_df() -> pd.DataFrame:
    """Carga el CSV disponible y garantiza columnas claves normalizadas."""
    csv_path = _pick_existing_path(CSV_CANDIDATES)
    df = cargar_datos(csv_path)
    print(f"✓ Usando archivo: {csv_path}")

    # Fechas
    if "start_date_time" not in df.columns:
        raise ValueError("Falta columna 'start_date_time'.")
    df["start_date_time"] = pd.to_datetime(df["start_date_time"], errors="coerce")
    df = df.dropna(subset=["start_date_time"])

    # Ids
    if "id" not in df.columns:
        if "transaction_id" in df.columns:
            df = df.rename(columns={"transaction_id": "id"})
        else:
            df = df.reset_index().rename(columns={"index": "id"})
    if "user_id" not in df.columns:
        raise ValueError("Falta columna 'user_id'.")

    # Estación
    if "evse_uid" not in df.columns:
        for cand in ["station_name", "evse_id", "charger_id"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "evse_uid"})
                break
    if "evse_uid" not in df.columns:
        raise ValueError("Falta columna de estación ('evse_uid').")

    # Monetario (COP reales)
    if "ingresos_cop" in df.columns:
        df["ingresos_cop"] = pd.to_numeric(df["ingresos_cop"], errors="coerce")
    elif "amount_transaction_cop" in df.columns:
        df["ingresos_cop"] = pd.to_numeric(df["amount_transaction_cop"], errors="coerce")
    elif "amount_transaction" in df.columns:
        # amount_transaction suele venir en CENTAVOS → dividir entre 100
        df["ingresos_cop"] = pd.to_numeric(df["amount_transaction"], errors="coerce") / 100.0
    else:
        raise ValueError("No encontré columna monetaria ('ingresos_cop' / 'amount_transaction_cop' / 'amount_transaction').")

    # Energía
    if "energy_kwh" in df.columns:
        df["energy_kwh"] = pd.to_numeric(df["energy_kwh"], errors="coerce")
    else:
        df["energy_kwh"] = np.nan

    return df

# -------------------------------------------------------------
# CLASIFICACIÓN DE CONECTOR
# -------------------------------------------------------------
def clasificar_conector(nombre: str) -> str:
    """Clasifica T1 / T2 / CCS / Otro a partir de evse_uid (heurística prudente)."""
    n = str(nombre).lower()
    if "_t1" in n or " t1" in n or "t101" in n:
        return "T1"
    if "_t2" in n or " t2" in n or "t201" in n or "t202" in n:
        return "T2"
    if "ccs" in n:
        return "CCS"
    return "Otro"

# -------------------------------------------------------------
# AGREGADOS
# -------------------------------------------------------------
def preparar_datos_comparacion(df: pd.DataFrame):
    """Agrega métricas por tipo de conector usando COP y kWh correctos."""
    df = df.copy()
    df["tipo_conector"] = df["evse_uid"].apply(clasificar_conector)

    metricas_tipo = (
        df.groupby("tipo_conector")
        .agg(
            total_transacciones=("id", "count"),
            usuarios_unicos=("user_id", "nunique"),
            estaciones=("evse_uid", "nunique"),
            ingresos_totales=("ingresos_cop", "sum"),
            ingreso_promedio=("ingresos_cop", "mean"),
            ingreso_mediano=("ingresos_cop", "median"),
            energia_total_kwh=("energy_kwh", "sum"),
            energia_promedio_kwh=("energy_kwh", "mean"),
        )
        .reset_index()
    )

    metricas_tipo["trans_por_estacion"] = metricas_tipo.apply(
        lambda r: (r["total_transacciones"] / r["estaciones"]) if r["estaciones"] else 0.0, axis=1
    )
    metricas_tipo["ingresos_por_estacion"] = metricas_tipo.apply(
        lambda r: (r["ingresos_totales"] / r["estaciones"]) if r["estaciones"] else 0.0, axis=1
    )
    metricas_tipo["usuarios_por_estacion"] = metricas_tipo.apply(
        lambda r: (r["usuarios_unicos"] / r["estaciones"]) if r["estaciones"] else 0.0, axis=1
    )
    metricas_tipo["ingresos_por_usuario"] = metricas_tipo.apply(
        lambda r: (r["ingresos_totales"] / r["usuarios_unicos"]) if r["usuarios_unicos"] else 0.0, axis=1
    )

    return metricas_tipo, df

def preparar_datos_por_estacion(df: pd.DataFrame):
    """Métricas por estación individual (COP, kWh válidos)."""
    estaciones = (
        df.groupby(["evse_uid", "tipo_conector"])
        .agg(
            transacciones=("id", "count"),
            usuarios=("user_id", "nunique"),
            ingresos=("ingresos_cop", "sum"),
            energia_kwh=("energy_kwh", "sum"),
        )
        .reset_index()
    )
    return estaciones

# -------------------------------------------------------------
# GRÁFICO PRINCIPAL
# -------------------------------------------------------------
def crear_grafico_principal(metricas_tipo: pd.DataFrame) -> go.Figure:
    t1_t2 = metricas_tipo[metricas_tipo["tipo_conector"].isin(["T1", "T2"])].copy()
    colors = {"T1": "#3b82f6", "T2": "#10b981"}

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Transacciones Totales",
            "Ingresos Totales",
            "Usuarios Únicos",
            "Energía Consumida (kWh)",
            "Transacciones por Estación",
            "Ingresos por Estación",
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.15
    )

    # 1. Transacciones
    fig.add_trace(
        go.Bar(
            x=t1_t2["tipo_conector"],
            y=t1_t2["total_transacciones"],
            marker_color=[colors[t] for t in t1_t2["tipo_conector"]],
            text=t1_t2["total_transacciones"],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Transacciones: %{y:,}<extra></extra>",
            showlegend=False,
            name="Transacciones",
        ),
        row=1, col=1
    )

    # 2. Ingresos (COP; etiqueta en millones)
    fig.add_trace(
        go.Bar(
            x=t1_t2["tipo_conector"],
            y=t1_t2["ingresos_totales"],
            marker_color=[colors[t] for t in t1_t2["tipo_conector"]],
            text=[f"${v/1e6:.1f}M" for v in t1_t2["ingresos_totales"]],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Ingresos: $%{y:,.0f} COP<extra></extra>",
            showlegend=False,
            name="Ingresos",
        ),
        row=1, col=2
    )

    # 3. Usuarios
    fig.add_trace(
        go.Bar(
            x=t1_t2["tipo_conector"],
            y=t1_t2["usuarios_unicos"],
            marker_color=[colors[t] for t in t1_t2["tipo_conector"]],
            text=t1_t2["usuarios_unicos"],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Usuarios: %{y:,}<extra></extra>",
            showlegend=False,
            name="Usuarios",
        ),
        row=2, col=1
    )

    # 4. Energía
    fig.add_trace(
        go.Bar(
            x=t1_t2["tipo_conector"],
            y=t1_t2["energia_total_kwh"],
            marker_color=[colors[t] for t in t1_t2["tipo_conector"]],
            text=[f"{v:,.0f} kWh" for v in t1_t2["energia_total_kwh"]],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Energía: %{y:,.0f} kWh<extra></extra>",
            showlegend=False,
            name="Energía",
        ),
        row=2, col=2
    )

    # 5. Trans/Estación
    fig.add_trace(
        go.Bar(
            x=t1_t2["tipo_conector"],
            y=t1_t2["trans_por_estacion"],
            marker_color=[colors[t] for t in t1_t2["tipo_conector"]],
            text=[f"{v:.0f}" for v in t1_t2["trans_por_estacion"]],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Trans/Estación: %{y:.1f}<extra></extra>",
            showlegend=False,
            name="Trans/Estación",
        ),
        row=3, col=1
    )

    # 6. $/Estación
    fig.add_trace(
        go.Bar(
            x=t1_t2["tipo_conector"],
            y=t1_t2["ingresos_por_estacion"],
            marker_color=[colors[t] for t in t1_t2["tipo_conector"]],
            text=[f"${v/1e6:.1f}M" for v in t1_t2["ingresos_por_estacion"]],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Ingresos/Estación: $%{y:,.0f} COP<extra></extra>",
            showlegend=False,
            name="$/Estación",
        ),
        row=3, col=2
    )

    fig.update_layout(
        title=dict(
            text="Comparación de Conectores: Tipo 1 (T1) vs Tipo 2 (T2)",
            x=0.5, xanchor="center",
            font=dict(size=22, color="#2d3748", family="Arial Black")
        ),
        showlegend=False,
        height=1200,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=12),
        margin=dict(l=70, r=30, t=80, b=60)
    )
    for i in range(1, 4):
        for j in range(1, 2+1):
            fig.update_xaxes(gridcolor="#e2e8f0", row=i, col=j)
            fig.update_yaxes(gridcolor="#e2e8f0", row=i, col=j)

    return fig

# -------------------------------------------------------------
# INSIGHTS Y RECONCILIACIONES
# -------------------------------------------------------------
def analizar_t1_vs_t2(metricas_tipo: pd.DataFrame, df: pd.DataFrame) -> dict:
    t1 = metricas_tipo[metricas_tipo["tipo_conector"] == "T1"].iloc[0] if (metricas_tipo["tipo_conector"] == "T1").any() else None
    t2 = metricas_tipo[metricas_tipo["tipo_conector"] == "T2"].iloc[0] if (metricas_tipo["tipo_conector"] == "T2").any() else None

    def val(row, key, default=0):
        return (row[key] if row is not None and key in row else default)

    def pct(a, b):
        return (b - a) / a * 100 if a else np.nan

    diff_trans_pct = pct(val(t1, "total_transacciones"), val(t2, "total_transacciones"))
    diff_ingresos_pct = pct(val(t1, "ingresos_totales"), val(t2, "ingresos_totales"))
    diff_usuarios_pct = pct(val(t1, "usuarios_unicos"), val(t2, "usuarios_unicos"))

    insights = {
        "t1_transacciones": int(val(t1, "total_transacciones")),
        "t1_ingresos": int(val(t1, "ingresos_totales")),
        "t1_usuarios": int(val(t1, "usuarios_unicos")),
        "t1_estaciones": int(val(t1, "estaciones")),
        "t1_energia": float(val(t1, "energia_total_kwh")),
        "t1_trans_por_estacion": float(val(t1, "trans_por_estacion")),
        "t1_ingresos_por_estacion": float(val(t1, "ingresos_por_estacion")),
        "t1_ingreso_promedio": float(val(t1, "ingreso_promedio")),

        "t2_transacciones": int(val(t2, "total_transacciones")),
        "t2_ingresos": int(val(t2, "ingresos_totales")),
        "t2_usuarios": int(val(t2, "usuarios_unicos")),
        "t2_estaciones": int(val(t2, "estaciones")),
        "t2_energia": float(val(t2, "energia_total_kwh")),
        "t2_trans_por_estacion": float(val(t2, "trans_por_estacion")),
        "t2_ingresos_por_estacion": float(val(t2, "ingresos_por_estacion")),
        "t2_ingreso_promedio": float(val(t2, "ingreso_promedio")),

        "diff_trans_pct": float(diff_trans_pct) if not pd.isna(diff_trans_pct) else 0.0,
        "diff_ingresos_pct": float(diff_ingresos_pct) if not pd.isna(diff_ingresos_pct) else 0.0,
        "diff_usuarios_pct": float(diff_usuarios_pct) if not pd.isna(diff_usuarios_pct) else 0.0,

        "ganador_trans": "T2" if val(t2, "total_transacciones") > val(t1, "total_transacciones") else "T1",
        "ganador_ingresos": "T2" if val(t2, "ingresos_totales") > val(t1, "ingresos_totales") else "T1",
        "ganador_eficiencia": "T2" if val(t2, "trans_por_estacion") > val(t1, "trans_por_estacion") else "T1",

        # Reconciliaciones
        "total_ingresos_cop": float(df["ingresos_cop"].sum()),
        "total_ingresos_t1_t2": float(metricas_tipo[metricas_tipo["tipo_conector"].isin(["T1","T2"])]["ingresos_totales"].sum()),
        "total_energia_kwh": float(df["energy_kwh"].sum(skipna=True)),
    }

    insights["insight_volumen"] = (
        f"T2 supera a T1 en {insights['diff_trans_pct']:.1f}% en transacciones totales "
        f"({insights['t2_transacciones']:,} vs {insights['t1_transacciones']:,})."
    )
    insights["insight_ingresos"] = (
        f"T2 genera {insights['diff_ingresos_pct']:.1f}% más ingresos que T1 "
        f"(${insights['t2_ingresos']:,} vs ${insights['t1_ingresos']:,} COP)."
    )
    insights["insight_eficiencia"] = (
        f"T2 es más eficiente con {insights['t2_trans_por_estacion']:.0f} trans/estación "
        f"vs {insights['t1_trans_por_estacion']:.0f} de T1."
    )
    insights["insight_usuarios"] = (
        f"T2 atrae {insights['diff_usuarios_pct']:.1f}% más usuarios únicos "
        f"({insights['t2_usuarios']} vs {insights['t1_usuarios']})."
    )
    return insights

# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "="*80)
    print("GRÁFICO 9: COMPARACIÓN T1 VS T2")
    print("="*80)

    # 1) Cargar datos normalizados
    df = cargar_df()
    print(f"✓ Registros: {len(df):,}")
    print(f" Total ingresos (COP): {df['ingresos_cop'].sum():,.0f}")
    if df["energy_kwh"].notna().any():
        print(f"⚡ Total energía (kWh): {df['energy_kwh'].sum(skipna=True):,.1f}")
    print()

    # 2) Agregados
    print("Clasificando conectores y preparando métricas…")
    metricas_tipo, df_clasificado = preparar_datos_comparacion(df)
    _ = preparar_datos_por_estacion(df_clasificado)  # (se mantiene por si lo necesitas después)
    print("✓ Métricas listas\n")

    # 3) Gráfico principal
    print("Creando gráfico principal…")
    fig_main = crear_grafico_principal(metricas_tipo)

    # 4) Insights + Reconciliación
    print("\n" + "="*80)
    print("ANÁLISIS COMPARATIVO T1 VS T2")
    print("="*80)
    insights = analizar_t1_vs_t2(metricas_tipo, df_clasificado)

    print(f"\n TIPO 1 (T1):")
    print(f"   • Estaciones: {insights['t1_estaciones']}")
    print(f"   • Transacciones: {insights['t1_transacciones']:,}")
    print(f"   • Ingresos: ${insights['t1_ingresos']:,} COP")
    print(f"   • Usuarios: {insights['t1_usuarios']:,}")
    print(f"   • Energía: {insights['t1_energia']:,} kWh")
    print(f"   • Trans/Estación: {insights['t1_trans_por_estacion']:.0f}")
    print(f"   • Ingreso promedio/trans: ${insights['t1_ingreso_promedio']:,.0f} COP")

    print(f"\n TIPO 2 (T2):")
    print(f"   • Estaciones: {insights['t2_estaciones']}")
    print(f"   • Transacciones: {insights['t2_transacciones']:,}")
    print(f"   • Ingresos: ${insights['t2_ingresos']:,} COP")
    print(f"   • Usuarios: {insights['t2_usuarios']:,}")
    print(f"   • Energía: {insights['t2_energia']:,} kWh")
    print(f"   • Trans/Estación: {insights['t2_trans_por_estacion']:.0f}")
    print(f"   • Ingreso promedio/trans: ${insights['t2_ingreso_promedio']:,.0f} COP")

    print(f"\n DIFERENCIAS:")
    print(f"   • Transacciones: T2 vs T1 = {insights['diff_trans_pct']:+.1f}%")
    print(f"   • Ingresos: T2 vs T1 = {insights['diff_ingresos_pct']:+.1f}%")
    print(f"   • Usuarios: T2 vs T1 = {insights['diff_usuarios_pct']:+.1f}%")

    print(f"\n GANADORES:")
    print(f"   • Volumen (transacciones): {insights['ganador_trans']}")
    print(f"   • Ingresos totales: {insights['ganador_ingresos']}")
    print(f"   • Eficiencia (trans/estación): {insights['ganador_eficiencia']}")

    # Reconciliación final
    print("\n Reconciliación:")
    print(f"   • Total COP (dataset): ${insights['total_ingresos_cop']:,.0f}")
    print(f"   • Suma T1+T2 (COP):    ${insights['total_ingresos_t1_t2']:,.0f}  "
          f"(el resto pertenece a CCS/Otro)")
    if df["energy_kwh"].notna().any():
        print(f"   • Total energía (kWh):  {insights['total_energia_kwh']:,.1f}")

    # 5) Guardar y mostrar
    print("\n" + "="*80)
    print("Guardando gráfico principal…")
    guardar_grafico(fig_main, "grafico_09_comparacion_t1_t2_principal.png")
    print("Abriendo gráfico principal…")
    fig_main.show()
