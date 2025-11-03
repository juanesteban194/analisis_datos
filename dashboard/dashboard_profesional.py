# -*- coding: utf-8 -*-
"""
OASIS EVSE - Dashboard Profesional v3.4 CORREGIDO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Pestañas superiores
- Sin emojis en interfaz visual
- Manejo robusto de errores y argumentos
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Imports básicos
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output
import plotly.graph_objects as go

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIGURACIÓN DE RUTAS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE) if os.path.basename(HERE) == "dashboard" else HERE

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

GRAFICOS_DIR = os.path.join(PROJECT_ROOT, "graficos")
if os.path.isdir(GRAFICOS_DIR) and GRAFICOS_DIR not in sys.path:
    sys.path.insert(0, GRAFICOS_DIR)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CARGAR DATOS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATA_PATHS = [
    os.path.join(PROJECT_ROOT, "data", "df_oasis_clean_normalized.csv"),
    os.path.join(PROJECT_ROOT, "data", "df_oasis_clean.csv"),
    os.path.join(PROJECT_ROOT, "data", "df_oasis_ready.csv"),
]

def cargar_dataset():
    """Carga el dataset principal"""
    for path in DATA_PATHS:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                
                if "start_date_time" in df.columns:
                    df["start_date_time"] = pd.to_datetime(df["start_date_time"], errors="coerce")
                    df = df.dropna(subset=["start_date_time"])
                
                if "ingresos_cop" not in df.columns and "amount_transaction" in df.columns:
                    df["ingresos_cop"] = pd.to_numeric(df["amount_transaction"], errors="coerce") / 100.0
                
                if "energy_kwh" not in df.columns and "energy_wh" in df.columns:
                    df["energy_kwh"] = pd.to_numeric(df["energy_wh"], errors="coerce") / 1000.0
                
                if "evse_uid" not in df.columns:
                    for col in ["station_name", "evse_id", "charger_id"]:
                        if col in df.columns:
                            df["evse_uid"] = df[col]
                            break
                
                print(f"✅ Dataset cargado: {len(df):,} registros desde {os.path.basename(path)}")
                return df
                
            except Exception as e:
                print(f"❌ Error cargando {path}: {e}")
                continue
    
    raise FileNotFoundError("No se encontró ningún CSV válido en data/")

try:
    DF_FULL = cargar_dataset()
except Exception as e:
    print(f"❌ ERROR CRÍTICO: {e}")
    sys.exit(1)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIGURACIÓN DE GRÁFICOS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GRAFICOS_CONFIG = {
    "overview": {
        "nombre": "Visión General",
        "color": "#667eea",
        "graficos": [
            {"id": "g01", "titulo": "Transacciones por Estación", "modulo": "grafico_01_barras_estaciones", "funcion": "crear_grafico"},
            {"id": "g05", "titulo": "Ingresos Mensuales", "modulo": "grafico_05_ingresos_mensuales", "funcion": "crear_grafico"}
        ]
    },
    "usuarios": {
        "nombre": "Usuarios",
        "color": "#10b981",
        "graficos": [
            {"id": "g02", "titulo": "Distribución de Usuarios", "modulo": "grafico_02_distribucion_usuarios", "funcion": "crear_grafico"},
            {"id": "g03", "titulo": "Segmentación RFM", "modulo": "grafico_03_segmentacion_rfm", "funcion": "crear_grafico_2d", "rfm": True},
            {"id": "g04", "titulo": "CLV por Segmentos", "modulo": "grafico_04_clv_segmentos", "funcion": "crear_grafico"},
            {"id": "g11", "titulo": "Retención de Usuarios", "modulo": "grafico_11_retencion_usuarios", "funcion": "crear_grafico", "cohortes": True}
        ]
    },
    "patrones": {
        "nombre": "Patrones Temporales",
        "color": "#f59e0b",
        "graficos": [
            {"id": "g06", "titulo": "Patrones Horarios", "modulo": "grafico_06_patrones_horarios", "funcion": "crear_grafico", "temporal": True},
            {"id": "g07", "titulo": "Heatmap de Uso", "modulo": "grafico_07_heatmap_uso", "funcion": "crear_grafico", "uso": True}
        ]
    },
    "estaciones": {
        "nombre": "Estaciones",
        "color": "#3b82f6",
        "graficos": [
            {"id": "g08", "titulo": "Top Estaciones", "modulo": "grafico_08_top_estaciones", "funcion": "figura_barras_top", "metricas": True},
            {"id": "g09", "titulo": "Comparación T1 vs T2", "modulo": "grafico_09_comparacion_t1_t2", "funcion": "crear_grafico_principal", "tipos": True},
            {"id": "g10", "titulo": "Energía por Estación", "modulo": "grafico_10_energia_estaciones", "funcion": "crear_grafico", "energia": True}
        ]
    }
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# INICIALIZAR APP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FUNCIONES DE UI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def crear_kpi(titulo, valor, subtitulo="", color="#667eea"):
    """Tarjeta KPI sin emojis"""
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.H6(titulo, className="text-muted", style={"fontSize": "11px", "marginBottom": "4px", "textTransform": "uppercase"}),
                html.H3(valor, style={"marginBottom": "0", "fontWeight": "bold", "color": color}),
                html.Small(subtitulo, className="text-muted")
            ])
        ], style={"padding": "20px"})
    ], style={"borderLeft": f"4px solid {color}", "marginBottom": "15px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"})

def crear_tabs():
    """Crea las pestañas principales"""
    tabs = []
    
    for cat_id, config in GRAFICOS_CONFIG.items():
        botones = []
        for graf in config["graficos"]:
            botones.append(
                dbc.Button(
                    graf["titulo"],
                    id={"type": "graf-btn", "cat": cat_id, "graf": graf["id"]},
                    color="light",
                    className="m-1",
                    style={"fontSize": "14px"}
                )
            )
        
        tab_content = html.Div([
            html.Div(botones, className="mb-3"),
            html.Div(id=f"content-{cat_id}")
        ], style={"padding": "20px"})
        
        tabs.append(
            dbc.Tab(
                tab_content,
                label=config["nombre"],
                tab_id=cat_id,
                tab_style={"marginLeft": "2px"},
                active_tab_style={"backgroundColor": config["color"], "color": "white", "fontWeight": "bold"}
            )
        )
    
    return dbc.Tabs(tabs, id="main-tabs", active_tab="overview")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CARGAR GRÁFICOS DINÁMICAMENTE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def cargar_grafico(config, df):
    """Carga un gráfico dinámicamente con manejo robusto de errores"""
    try:
        modulo = __import__(config["modulo"])
        
        # Casos especiales
        if config.get("rfm"):
            try:
                rfm = modulo.calcular_rfm(df)
                func = getattr(modulo, config["funcion"])
                return func(rfm)
            except Exception as e:
                raise Exception(f"Error en RFM: {str(e)}")
                
        elif config.get("cohortes"):
            try:
                ret_matrix, cohort_sizes, avg_curve, _ = modulo.preparar_cohortes(df)
                stats = modulo.calcular_one_timers_repeaters(df)
                return modulo.crear_grafico(ret_matrix, avg_curve, stats)
            except Exception as e:
                raise Exception(f"Error en cohortes: {str(e)}")
                
        elif config.get("temporal"):
            # Gráfico 06: Patrones Horarios - necesita preparar datos Y matriz
            try:
                df_t = modulo.preparar_datos_temporales(df)
                matriz = modulo.crear_matriz_heatmap(df_t)
                return modulo.crear_grafico(df_t, matriz)
            except Exception as e:
                raise Exception(f"Error en temporal: {str(e)}")
                
        elif config.get("uso"):
            # Gráfico 07: Heatmap de Uso
            try:
                uso = modulo.preparar_datos_uso(df)
                matriz = modulo.crear_matriz_temporal(df, uso["evse_uid"].tolist())
                return modulo.crear_grafico(uso, matriz)
            except Exception as e:
                raise Exception(f"Error en uso: {str(e)}")
                
        elif config.get("metricas"):
            # Gráfico 08: Top estaciones
            try:
                metricas = modulo.calcular_metricas_estaciones(df)
                func = getattr(modulo, config["funcion"])
                return func(metricas, top_n=20)
            except Exception as e:
                raise Exception(f"Error en métricas: {str(e)}")
                
        elif config.get("tipos"):
            # Gráfico 09: T1 vs T2
            try:
                metricas_tipo, _ = modulo.preparar_datos_comparacion(df)
                func = getattr(modulo, config["funcion"])
                return func(metricas_tipo)
            except Exception as e:
                raise Exception(f"Error en tipos: {str(e)}")
                
        elif config.get("energia"):
            # Gráfico 10: Energía - necesita preparar y crear matriz
            try:
                agg_est = modulo.preparar_metricas_estacion(df)
                matriz = modulo.crear_matriz_temporal_kwh(df)
                return modulo.crear_grafico(agg_est, matriz)
            except Exception as e:
                raise Exception(f"Error en energía: {str(e)}")
        else:
            # Caso estándar: solo pasar df
            func = getattr(modulo, config["funcion"])
            return func(df)
            
    except Exception as e:
        # Crear figura de error detallada
        fig = go.Figure()
        error_msg = f"Error cargando {config['titulo']}: {str(e)}"
        print(f"❌ {error_msg}")
        
        fig.add_annotation(
            text=f"⚠ {error_msg[:150]}...",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="#ef4444"),
            align="center"
        )
        fig.update_layout(
            title=config["titulo"],
            height=500,
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
        return fig

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LAYOUT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

app.layout = html.Div([
    # Header sin emojis
    html.Div([
        html.Div([
            html.Span("OASIS EVSE Analytics", style={"fontWeight": "bold", "fontSize": "24px", "color": "white"}),
            html.Span(
                f"  |  {len(DF_FULL):,} transacciones  |  {DF_FULL['user_id'].nunique():,} usuarios", 
                style={"marginLeft": "20px", "fontSize": "14px", "color": "rgba(255,255,255,0.8)"}
            )
        ])
    ], style={
        "height": "70px",
        "backgroundColor": "#343a40",
        "padding": "0 30px",
        "display": "flex",
        "alignItems": "center",
        "boxShadow": "0 2px 10px rgba(0,0,0,0.1)"
    }),
    
    # Contenido principal
    html.Div([
        # KPIs sin emojis
        dbc.Row([
            dbc.Col(crear_kpi(
                "Ingresos Totales",
                f"${DF_FULL['ingresos_cop'].sum()/1e6:.1f}M COP",
                f"Ticket promedio: ${DF_FULL['ingresos_cop'].mean():,.0f}",
                "#10b981"
            ), width=3),
            dbc.Col(crear_kpi(
                "Transacciones",
                f"{len(DF_FULL):,}",
                f"Promedio diario: {len(DF_FULL)/365:.0f}",
                "#3b82f6"
            ), width=3),
            dbc.Col(crear_kpi(
                "Usuarios Únicos",
                f"{DF_FULL['user_id'].nunique():,}",
                f"{len(DF_FULL)/DF_FULL['user_id'].nunique():.1f} transacciones/usuario",
                "#8b5cf6"
            ), width=3),
            dbc.Col(crear_kpi(
                "Estaciones Activas",
                f"{DF_FULL.get('evse_uid', pd.Series()).nunique():,}",
                f"{len(DF_FULL)/DF_FULL.get('evse_uid', pd.Series()).nunique():.0f} transacciones/estación",
                "#f59e0b"
            ), width=3),
        ], style={"marginTop": "20px", "marginBottom": "20px"}),
        
        # Pestañas
        crear_tabs()
        
    ], style={"padding": "20px", "maxWidth": "1400px", "margin": "0 auto"})
    
], style={"backgroundColor": "#f8f9fa", "minHeight": "100vh"})

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CALLBACKS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Callback para cada categoría
for cat_id in GRAFICOS_CONFIG.keys():
    @app.callback(
        Output(f"content-{cat_id}", "children"),
        Input({"type": "graf-btn", "cat": cat_id, "graf": dash.dependencies.ALL}, "n_clicks"),
        prevent_initial_call=True
    )
    def mostrar_grafico_cat(n_clicks, cat=cat_id):
        ctx = dash.callback_context
        if not ctx.triggered or not any(n_clicks):
            return html.Div()
        
        # Obtener ID del botón clickeado
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        import json
        graf_data = json.loads(button_id)
        graf_id = graf_data["graf"]
        
        # Buscar configuración
        config = None
        for graf in GRAFICOS_CONFIG[cat]["graficos"]:
            if graf["id"] == graf_id:
                config = graf
                break
        
        if not config:
            return html.Div("Gráfico no encontrado", className="alert alert-danger")
        
        # Cargar gráfico
        print(f"📊 Cargando gráfico: {config['titulo']}")
        fig = cargar_grafico(config, DF_FULL)
        
        return dbc.Card([
            dbc.CardHeader([
                html.H5(config["titulo"], style={"marginBottom": "0", "color": "#2d3748"})
            ], style={"backgroundColor": "#f8f9fa", "borderBottom": "2px solid #e2e8f0"}),
            dbc.CardBody([
                dcc.Graph(
                    figure=fig,
                    config={"displayModeBar": True, "displaylogo": False},
                    style={"height": "650px"}
                )
            ])
        ], style={"marginTop": "15px", "boxShadow": "0 4px 6px rgba(0,0,0,0.1)"})

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RUN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 OASIS EVSE DASHBOARD v3.4 - CORREGIDO")
    print("="*60)
    print(f"📊 Dataset: {len(DF_FULL):,} registros")
    print(f"👥 Usuarios: {DF_FULL['user_id'].nunique():,}")
    print(f"🌐 URL: http://127.0.0.1:8050")
    print("="*60 + "\n")
    
    app.run_server(debug=True, port=8050, host="127.0.0.1")