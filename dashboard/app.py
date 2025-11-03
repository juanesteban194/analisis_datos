# -*- coding: utf-8 -*-
"""
Dashboard Oasis — robusto contra rutas (soporta ejecutarlo desde / o /dashboard/)
- Autodetección de raíz del proyecto (busca utils.py y data/)
- Carga segura de datos (fallback local si no existe utils.cargar_datos)
- Integra gráficos 01..11 (ajusta nombres si alguno difiere)
"""

import os, sys
import pandas as pd
import numpy as np
from datetime import datetime

from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

# ---------------------------------------------------------------------
# Ubicación robusta de la raíz del proyecto (sube hasta encontrar utils.py)
# ---------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = [HERE, os.path.dirname(HERE), os.path.dirname(os.path.dirname(HERE))]

PROJECT_ROOT = None
for base in CANDIDATES:
    if os.path.exists(os.path.join(base, "utils.py")) and os.path.isdir(os.path.join(base, "data")):
        PROJECT_ROOT = base
        break
if PROJECT_ROOT is None:
    # última opción: asume 2 niveles arriba
    PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))

# Asegurar paths para imports
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
GRAFICOS_DIR = os.path.join(PROJECT_ROOT, "graficos")
if os.path.isdir(GRAFICOS_DIR) and GRAFICOS_DIR not in sys.path:
    sys.path.insert(0, GRAFICOS_DIR)

# ---------------------------------------------------------------------
# Loader de datos: usa utils.cargar_datos si existe, si no, fallback local
# ---------------------------------------------------------------------
def cargar_datos_local(csv_path: str) -> pd.DataFrame:
    parse_cands = [
        "start_date_time", "start_datetime", "start_time",
        "created_at", "timestamp"
    ]
    df = pd.read_csv(csv_path)
    for c in parse_cands:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

try:
    from utils import cargar_datos as _cargar_datos
except Exception:
    _cargar_datos = cargar_datos_local  # fallback

# Candidatos de datos
DATA_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "data", "df_oasis_clean_normalized.csv"),
    os.path.join(PROJECT_ROOT, "data", "df_oasis_clean.csv"),
    os.path.join(PROJECT_ROOT, "data", "df_oasis_ready.csv"),
]

def _pick_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("No se encontró CSV en:\n- " + "\n- ".join(paths))

def load_df():
    csv = _pick_existing(DATA_CANDIDATES)
    df = _cargar_datos(csv)

    # Fechas
    fecha_col = "start_date_time" if "start_date_time" in df.columns else None
    if fecha_col is None:
        for c in ["start_datetime", "timestamp", "created_at"]:
            if c in df.columns:
                fecha_col = c; break
    if fecha_col:
        df[fecha_col] = pd.to_datetime(df[fecha_col], errors="coerce")
        df = df.dropna(subset=[fecha_col]).rename(columns={fecha_col: "start_date_time"})
    else:
        raise ValueError("No se encontró columna de fecha compatible.")

    # Ingresos en COP (regla que cuadra ~166M): amount_transaction / 100
    if "ingresos_cop" in df.columns:
        df["ingresos_cop"] = pd.to_numeric(df["ingresos_cop"], errors="coerce")
    else:
        df["ingresos_cop"] = pd.to_numeric(df.get("amount_transaction", np.nan), errors="coerce") / 100.0

    # Energía kWh
    if "energy_kwh" in df.columns:
        df["energy_kwh"] = pd.to_numeric(df["energy_kwh"], errors="coerce")
    elif "energy_wh" in df.columns:
        df["energy_kwh"] = pd.to_numeric(df["energy_wh"], errors="coerce") / 1000.0
    elif "energy" in df.columns:
        s = pd.to_numeric(df["energy"], errors="coerce")
        df["energy_kwh"] = s/1000.0 if s.max(skipna=True) and s.max() > 2000 else s
    else:
        df["energy_kwh"] = np.nan

    # Estación
    if "evse_uid" not in df.columns:
        for c in ["station_name", "evse_id", "charger_id"]:
            if c in df.columns:
                df = df.rename(columns={c: "evse_uid"})
                break

    return df

DF_FULL = load_df()

# ---------------------------------------------------------------------
# Figuras: import seguro de tus módulos, con fallback simple si falta alguno
# ---------------------------------------------------------------------
def safe_fig(title, subtitle):
    fig = go.Figure()
    fig.update_layout(title=f"{title} — {subtitle}", paper_bgcolor="white", plot_bgcolor="white", height=420)
    return fig

def fig_g01(d):
    try:
        mod = __import__("grafico_01_barras_estaciones")
        return mod.crear_figura(d) if hasattr(mod, "crear_figura") else safe_fig("G01", "crear_figura() no existe")
    except Exception as e:
        # fallback: barras por ingresos
        agg = d.groupby("evse_uid")["ingresos_cop"].sum().sort_values(ascending=False).head(20)
        fig = go.Figure(go.Bar(x=agg.values/1e6, y=agg.index, orientation="h"))
        fig.update_layout(title="Top estaciones por ingresos (M COP)", height=600)
        return fig

def fig_g02(d):
    try:
        mod = __import__("grafico_02_distribucion_usuarios")
        return mod.crear_figura(d) if hasattr(mod, "crear_figura") else safe_fig("G02", "crear_figura() no existe")
    except Exception as e:
        g = d.groupby("user_id")["id"].count() if "user_id" in d.columns else pd.Series(dtype=int)
        fig = go.Figure(go.Histogram(x=g, nbinsx=30))
        fig.update_layout(title="Distribución: transacciones por usuario")
        return fig

def fig_g03(d):
    try:
        mod = __import__("grafico_03_segmentacion_rfm")
        if hasattr(mod, "calcular_rfm") and hasattr(mod, "crear_grafico_2d"):
            rfm = mod.calcular_rfm(d)
            return mod.crear_grafico_2d(rfm)
        return safe_fig("G03", "faltan funciones calcular_rfm/crear_grafico_2d")
    except Exception as e:
        return safe_fig("G03", str(e))

def fig_g04(d):
    try:
        mod = __import__("grafico_04_clv_segmentos")
        return mod.crear_figura(d) if hasattr(mod, "crear_figura") else safe_fig("G04", "crear_figura() no existe")
    except Exception as e:
        return safe_fig("G04", str(e))

def fig_g05(d):
    try:
        mod = __import__("grafico_05_ingresos_mensuales")
        return mod.crear_figura(d) if hasattr(mod, "crear_figura") else safe_fig("G05", "crear_figura() no existe")
    except Exception as e:
        m = d.assign(ym=d["start_date_time"].dt.to_period("M").astype(str)).groupby("ym")["ingresos_cop"].sum()
        fig = go.Figure(go.Scatter(x=m.index, y=m.values/1e6, mode="lines+markers"))
        fig.update_layout(title="Ingresos mensuales (M COP)")
        return fig

def fig_g06(d):
    try:
        mod = __import__("grafico_06_patrones_horarios")
        return mod.crear_figura(d) if hasattr(mod, "crear_figura") else safe_fig("G06", "crear_figura() no existe")
    except Exception as e:
        return safe_fig("G06", str(e))

def fig_g07(d):
    try:
        mod = __import__("grafico_07_heatmap_uso")
        uso = mod.preparar_datos_uso(d); mat = mod.crear_matriz_temporal(d)
        return mod.crear_grafico(uso, mat)
    except Exception as e:
        return safe_fig("G07", str(e))

def fig_g08_main(d):
    try:
        mod = __import__("grafico_08_top_estaciones")
        met = mod.calcular_metricas_estaciones(d)
        return mod.figura_barras_top(met, top_n=20)
    except Exception as e:
        return safe_fig("G08 Barras", str(e))

def fig_g08_radar(d):
    try:
        mod = __import__("grafico_08_top_estaciones")
        met = mod.calcular_metricas_estaciones(d)
        return mod.figura_radar(met, top_n=6)
    except Exception as e:
        return safe_fig("G08 Radar", str(e))

def fig_g09(d):
    try:
        mod = __import__("grafico_09_comparacion_t1_t2")
        mt, dfc = mod.preparar_datos_comparacion(d)
        return mod.crear_grafico_principal(mt)
    except Exception as e:
        return safe_fig("G09", str(e))

def fig_g10(d):
    try:
        mod = __import__("grafico_10_kpis")
        return mod.crear_figura(d) if hasattr(mod, "crear_figura") else safe_fig("G10", "crear_figura() no existe")
    except Exception as e:
        return safe_fig("G10", str(e))

def fig_g11(d):
    try:
        mod = __import__("grafico_11_retencion_usuarios")
        retention_matrix, cohort_sizes, avg_curve, _ = mod.preparar_cohortes(d)
        return mod.crear_grafico(retention_matrix, avg_curve, mod.calcular_one_timers_repeaters(d))
    except Exception as e:
        return safe_fig("G11", str(e))

# ---------------------------------------------------------------------
# Filtros + KPIs
# ---------------------------------------------------------------------
def clasificar_conector(evse):
    s = str(evse).lower()
    if "_t1" in s or "t101" in s: return "T1"
    if "_t2" in s or "t201" in s or "t202" in s: return "T2"
    if "ccs" in s: return "CCS"
    return "Otro"

def filter_df(df, date_range, station, connector):
    d = df.copy()
    if date_range and len(date_range) == 2:
        start = pd.to_datetime(date_range[0]); end = pd.to_datetime(date_range[1])
        d = d[(d["start_date_time"] >= start) & (d["start_date_time"] <= end)]
    if station and station != "TODAS":
        d = d[d["evse_uid"] == station]
    if connector and connector != "TODOS":
        d = d.assign(tipo_conector=d["evse_uid"].apply(clasificar_conector))
        d = d[d["tipo_conector"] == connector]
    return d

def kpi_cards(df):
    total_cop = df["ingresos_cop"].sum(skipna=True)
    total_kwh = df["energy_kwh"].sum(skipna=True)
    total_tx  = len(df)
    users     = df["user_id"].nunique() if "user_id" in df.columns else np.nan
    estaciones= df["evse_uid"].nunique() if "evse_uid" in df.columns else np.nan

    def card(t, v, sub=""):
        return dbc.Card(dbc.CardBody([html.Div(t, className="text-muted"),
                                      html.H3(v, className="mb-0"),
                                      html.Small(sub, className="text-muted")]), className="shadow-sm")
    return dbc.Row([
        dbc.Col(card("Ingresos", f"${total_cop:,.0f}", f"~{total_cop/1e6:.2f} M COP"), md=3),
        dbc.Col(card("Energía",  f"{(0 if pd.isna(total_kwh) else total_kwh):,.0f} kWh"), md=2),
        dbc.Col(card("Transacciones", f"{total_tx:,}"), md=2),
        dbc.Col(card("Usuarios únicos", f"{(0 if pd.isna(users) else users):,}"), md=2),
        dbc.Col(card("Estaciones", f"{(0 if pd.isna(estaciones) else estaciones):,}"), md=3),
    ], className="g-3")

# ---------------------------------------------------------------------
# App
# ---------------------------------------------------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.LUX], title="Oasis Dashboard")

min_date = DF_FULL["start_date_time"].min()
max_date = DF_FULL["start_date_time"].max()
stations = ["TODAS"] + (sorted(DF_FULL["evse_uid"].dropna().astype(str).unique().tolist()) if "evse_uid" in DF_FULL.columns else [])
connectors = ["TODOS", "T1", "T2", "CCS", "Otro"]

controls = dbc.Card(
    dbc.CardBody([
        html.H5("Filtros"),
        dbc.Row([
            dbc.Col([
                html.Label("Rango de fechas"),
                dcc.DatePickerRange(
                    id="f-fechas", min_date_allowed=min_date, max_date_allowed=max_date,
                    start_date=min_date, end_date=max_date, display_format="YYYY-MM-DD", style={"width":"100%"}
                )
            ], md=5),
            dbc.Col([
                html.Label("Estación"),
                dcc.Dropdown(id="f-estacion", options=[{"label": s, "value": s} for s in (stations or ["TODAS"])],
                             value="TODAS", clearable=False)
            ], md=4),
            dbc.Col([
                html.Label("Tipo conector"),
                dcc.Dropdown(id="f-conector", options=[{"label": c, "value": c} for c in connectors],
                             value="TODOS", clearable=False)
            ], md=3),
        ], className="g-2"),
    ]),
    className="mb-3 shadow-sm"
)

tabs = dcc.Tabs(id="tabs", value="tab-general", children=[
    dcc.Tab(label="Visión General", value="tab-general"),
    dcc.Tab(label="Usuarios & RFM", value="tab-usuarios"),
    dcc.Tab(label="Patrones de Uso", value="tab-patrones"),
    dcc.Tab(label="Estaciones & Conectores", value="tab-estaciones"),
    dcc.Tab(label="Cohortes & Retención", value="tab-cohortes"),
])

app.layout = dbc.Container([
    html.H2("Oasis — Tablero de Análisis", className="mt-3 mb-1"),
    html.Div("Normalización: ingresos = amount_transaction / 100 (COP)", className="text-muted mb-3"),
    controls,
    html.Div(id="kpis"),
    html.Hr(),
    tabs,
    html.Div(id="tab-content", className="mt-3 mb-5"),
], fluid=True)

@app.callback(
    Output("kpis", "children"),
    Output("tab-content", "children"),
    Input("f-fechas", "start_date"),
    Input("f-fechas", "end_date"),
    Input("f-estacion", "value"),
    Input("f-conector", "value"),
    Input("tabs", "value"),
)
def update_dashboard(start_date, end_date, estacion, conector, tab):
    dff = filter_df(DF_FULL, [start_date, end_date], estacion, conector)

    if tab == "tab-general":
        content = dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_g01(dff)), md=6),
            dbc.Col(dcc.Graph(figure=fig_g05(dff)), md=6),
        ], className="g-3")
    elif tab == "tab-usuarios":
        content = dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_g02(dff)), md=6),
            dbc.Col(dcc.Graph(figure=fig_g03(dff)), md=6),
            dbc.Col(dcc.Graph(figure=fig_g04(dff)), md=12),
        ], className="g-3")
    elif tab == "tab-patrones":
        content = dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_g06(dff)), md=6),
            dbc.Col(dcc.Graph(figure=fig_g07(dff)), md=6),
        ], className="g-3")
    elif tab == "tab-estaciones":
        content = dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_g08_main(dff)), md=7),
            dbc.Col(dcc.Graph(figure=fig_g08_radar(dff)), md=5),
            dbc.Col(dcc.Graph(figure=fig_g09(dff)), md=12),
        ], className="g-3")
    else:  # tab-cohortes
        content = dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_g10(dff)), md=12),
            dbc.Col(dcc.Graph(figure=fig_g11(dff)), md=12),
        ], className="g-3")

    return kpi_cards(dff), content

if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
