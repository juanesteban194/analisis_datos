# -*- coding: utf-8 -*-
"""
🎨 OASIS EVSE - Dashboard Profesional v3.2 SIMPLIFICADO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
100% Compatible - Sin errores de imports
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Imports básicos y seguros
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, MATCH, ALL
import plotly.graph_objects as go

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 📁 CONFIGURACIÓN DE RUTAS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE) if os.path.basename(HERE) == "dashboard" else HERE

# Agregar al path
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

GRAFICOS_DIR = os.path.join(PROJECT_ROOT, "graficos")
if os.path.isdir(GRAFICOS_DIR) and GRAFICOS_DIR not in sys.path:
    sys.path.insert(0, GRAFICOS_DIR)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 📊 CARGAR DATOS
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
                
                # Normalizar fechas
                if "start_date_time" in df.columns:
                    df["start_date_time"] = pd.to_datetime(df["start_date_time"], errors="coerce")
                    df = df.dropna(subset=["start_date_time"])
                
                # Ingresos
                if "ingresos_cop" not in df.columns and "amount_transaction" in df.columns:
                    df["ingresos_cop"] = pd.to_numeric(df["amount_transaction"], errors="coerce") / 100.0
                
                # Energía
                if "energy_kwh" not in df.columns and "energy_wh" in df.columns:
                    df["energy_kwh"] = pd.to_numeric(df["energy_wh"], errors="coerce") / 1000.0
                
                # Estación
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
    print("   Asegúrate de tener un CSV en la carpeta data/")
    sys.exit(1)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🎨 CONFIGURACIÓN DE GRÁFICOS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GRAFICOS_CONFIG = {
    "overview": {
        "nombre": "📊 Visión General",
        "icon": "📊",
        "color": "#667eea",
        "graficos": [
            {"id": "g01", "titulo": "Transacciones por Estación", "modulo": "grafico_01_barras_estaciones", "funcion": "crear_grafico"},
            {"id": "g05", "titulo": "Ingresos Mensuales", "modulo": "grafico_05_ingresos_mensuales", "funcion": "crear_grafico"}
        ]
    },
    "usuarios": {
        "nombre": "👥 Usuarios",
        "icon": "👥",
        "color": "#10b981",
        "graficos": [
            {"id": "g02", "titulo": "Distribución de Usuarios", "modulo": "grafico_02_distribucion_usuarios", "funcion": "crear_grafico"},
            {"id": "g03", "titulo": "Segmentación RFM", "modulo": "grafico_03_segmentacion_rfm", "funcion": "crear_grafico_2d", "rfm": True},
            {"id": "g04", "titulo": "CLV por Segmentos", "modulo": "grafico_04_clv_segmentos", "funcion": "crear_grafico"},
            {"id": "g11", "titulo": "Retención", "modulo": "grafico_11_retencion_usuarios", "funcion": "crear_grafico", "cohortes": True}
        ]
    },
    "patrones": {
        "nombre": "⏰ Patrones",
        "icon": "⏰",
        "color": "#f59e0b",
        "graficos": [
            {"id": "g06", "titulo": "Patrones Horarios", "modulo": "grafico_06_patrones_horarios", "funcion": "crear_grafico"},
            {"id": "g07", "titulo": "Heatmap de Uso", "modulo": "grafico_07_heatmap_uso", "funcion": "crear_grafico", "uso": True}
        ]
    },
    "estaciones": {
        "nombre": "🔌 Estaciones",
        "icon": "🔌",
        "color": "#3b82f6",
        "graficos": [
            {"id": "g08", "titulo": "Top Estaciones", "modulo": "grafico_08_top_estaciones", "funcion": "figura_barras_top", "metricas": True},
            {"id": "g09", "titulo": "T1 vs T2", "modulo": "grafico_09_comparacion_t1_t2", "funcion": "crear_grafico_principal", "tipos": True},
            {"id": "g10", "titulo": "Energía", "modulo": "grafico_10_energia_estaciones", "funcion": "crear_grafico"}
        ]
    }
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🎨 INICIALIZAR APP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
    ],
    suppress_callback_exceptions=True
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🎨 FUNCIONES DE UI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def crear_kpi(titulo, valor, subtitulo="", icon="📊", color="#667eea"):
    """Tarjeta KPI"""
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Div(icon, style={"fontSize": "24px", "marginBottom": "8px"}),
                html.H6(titulo, className="text-muted", style={"fontSize": "11px", "marginBottom": "4px"}),
                html.H3(valor, style={"marginBottom": "0", "fontWeight": "bold"}),
                html.Small(subtitulo, className="text-muted")
            ])
        ], style={"padding": "20px"})
    ], style={"borderLeft": f"4px solid {color}", "marginBottom": "15px"})

def crear_menu():
    """Menú lateral"""
    items = []
    for cat_id, config in GRAFICOS_CONFIG.items():
        # Header de categoría
        items.append(
            html.Div([
                html.Div([
                    html.Span(config["icon"], style={"marginRight": "8px"}),
                    html.Span(config["nombre"], style={"fontWeight": "600"}),
                    html.Span("▼", id=f"chevron-{cat_id}", style={"marginLeft": "auto", "fontSize": "12px"})
                ], 
                id=f"cat-{cat_id}",
                style={
                    "padding": "12px 16px",
                    "cursor": "pointer",
                    "display": "flex",
                    "alignItems": "center",
                    "backgroundColor": "#f8f9fa",
                    "borderLeft": f"4px solid {config['color']}",
                    "marginBottom": "8px"
                })
            ])
        )
        
        # Items de gráficos (inicialmente ocultos)
        grafico_items = []
        for graf in config["graficos"]:
            grafico_items.append(
                html.Div(
                    [html.Span("• ", style={"marginRight": "8px"}), html.Span(graf["titulo"])],
                    id={"type": "graf-item", "cat": cat_id, "graf": graf["id"]},
                    style={
                        "padding": "8px 16px 8px 32px",
                        "cursor": "pointer",
                        "fontSize": "14px",
                        "marginBottom": "4px"
                    },
                    className="menu-item"
                )
            )
        
        items.append(
            html.Div(
                grafico_items,
                id=f"items-{cat_id}",
                style={"display": "none"}  # Inicialmente oculto
            )
        )
    
    return html.Div([
        html.Div([
            html.H5("📊 Analytics", style={"color": "white", "margin": "0"})
        ], style={"padding": "16px", "backgroundColor": "#343a40"}),
        html.Div(items, style={"padding": "16px"})
    ], style={
        "width": "280px",
        "height": "100vh",
        "position": "fixed",
        "top": "60px",
        "left": "0",
        "backgroundColor": "white",
        "boxShadow": "2px 0 10px rgba(0,0,0,0.1)",
        "overflowY": "auto"
    })

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 📊 CARGAR GRÁFICOS DINÁMICAMENTE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def cargar_grafico(config, df):
    """Carga un gráfico dinámicamente"""
    try:
        modulo = __import__(config["modulo"])
        
        # Casos especiales
        if config.get("rfm"):
            rfm = modulo.calcular_rfm(df)
            func = getattr(modulo, config["funcion"])
            return func(rfm)
        elif config.get("cohortes"):
            ret_matrix, cohort_sizes, avg_curve, _ = modulo.preparar_cohortes(df)
            stats = modulo.calcular_one_timers_repeaters(df)
            return modulo.crear_grafico(ret_matrix, avg_curve, stats)
        elif config.get("uso"):
            uso = modulo.preparar_datos_uso(df)
            matriz = modulo.crear_matriz_temporal(df, uso["evse_uid"].tolist())
            return modulo.crear_grafico(uso, matriz)
        elif config.get("metricas"):
            metricas = modulo.calcular_metricas_estaciones(df)
            func = getattr(modulo, config["funcion"])
            return func(metricas, top_n=20)
        elif config.get("tipos"):
            metricas_tipo, _ = modulo.preparar_datos_comparacion(df)
            func = getattr(modulo, config["funcion"])
            return func(metricas_tipo)
        else:
            func = getattr(modulo, config["funcion"])
            return func(df)
            
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"⚠️ Error: {str(e)[:100]}",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color="#ef4444")
        )
        fig.update_layout(title=config["titulo"], height=500)
        return fig

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🎨 LAYOUT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

app.layout = html.Div([
    # Header
    html.Div([
        html.Div([
            html.Span("⚡ ", style={"fontSize": "24px"}),
            html.Span("OASIS EVSE", style={"fontWeight": "bold", "fontSize": "20px"}),
            html.Span(f"  |  📊 {len(DF_FULL):,} tx  |  👥 {DF_FULL['user_id'].nunique():,} usuarios", 
                     style={"marginLeft": "20px", "fontSize": "14px"})
        ], style={"color": "white"})
    ], style={
        "height": "60px",
        "backgroundColor": "#343a40",
        "padding": "0 20px",
        "display": "flex",
        "alignItems": "center",
        "position": "fixed",
        "top": "0",
        "left": "0",
        "right": "0",
        "zIndex": "1000"
    }),
    
    # Menu lateral
    crear_menu(),
    
    # Contenido principal
    html.Div([
        # KPIs
        dbc.Row([
            dbc.Col(crear_kpi("💰 Ingresos", f"${DF_FULL['ingresos_cop'].sum()/1e6:.1f}M", f"{DF_FULL['ingresos_cop'].sum():,.0f} COP", "💰", "#10b981"), width=3),
            dbc.Col(crear_kpi("📝 Transacciones", f"{len(DF_FULL):,}", f"Ticket: ${DF_FULL['ingresos_cop'].mean():,.0f}", "📝", "#3b82f6"), width=3),
            dbc.Col(crear_kpi("👥 Usuarios", f"{DF_FULL['user_id'].nunique():,}", f"{len(DF_FULL)/DF_FULL['user_id'].nunique():.1f} tx/usuario", "👥", "#8b5cf6"), width=3),
            dbc.Col(crear_kpi("🔌 Estaciones", f"{DF_FULL.get('evse_uid', pd.Series()).nunique():,}", f"{len(DF_FULL)/DF_FULL.get('evse_uid', pd.Series()).nunique():.0f} tx/est", "🔌", "#f59e0b"), width=3),
        ], style={"marginBottom": "20px"}),
        
        # Área de bienvenida/gráfico
        html.Div([
            html.Div([
                html.H2("Bienvenido a Oasis EVSE Analytics", style={"marginTop": "60px"}),
                html.P("Selecciona un gráfico del menú lateral", className="text-muted")
            ], id="welcome", style={"textAlign": "center", "padding": "60px"})
        ], id="content-area")
        
    ], style={"marginLeft": "280px", "marginTop": "60px", "padding": "20px"})
    
], style={"fontFamily": "Arial, sans-serif"})

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🔄 CALLBACKS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Toggle categorías
for cat_id in GRAFICOS_CONFIG.keys():
    @app.callback(
        [Output(f"items-{cat_id}", "style"),
         Output(f"chevron-{cat_id}", "children")],
        Input(f"cat-{cat_id}", "n_clicks"),
        State(f"items-{cat_id}", "style"),
        prevent_initial_call=True
    )
    def toggle_categoria(n, current_style):
        if n and current_style.get("display") == "none":
            return {"display": "block"}, "▲"
        return {"display": "none"}, "▼"

# Mostrar gráfico seleccionado
@app.callback(
    Output("content-area", "children"),
    Input({"type": "graf-item", "cat": ALL, "graf": ALL}, "n_clicks"),
    prevent_initial_call=True
)
def mostrar_grafico(n_clicks):
    ctx = dash.callback_context
    if not ctx.triggered or not any(n_clicks):
        return dash.no_update
    
    # Obtener ID del botón clickeado
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    import json
    graf_data = json.loads(button_id)
    graf_id = graf_data["graf"]
    
    # Buscar configuración
    config = None
    for cat in GRAFICOS_CONFIG.values():
        for graf in cat["graficos"]:
            if graf["id"] == graf_id:
                config = graf
                break
        if config:
            break
    
    if not config:
        return html.Div("Gráfico no encontrado")
    
    # Cargar gráfico
    fig = cargar_grafico(config, DF_FULL)
    
    return dbc.Card([
        dbc.CardHeader([
            html.H4(config["titulo"], style={"marginBottom": "0"})
        ]),
        dbc.CardBody([
            dcc.Graph(figure=fig, config={"displayModeBar": True}, style={"height": "600px"})
        ])
    ])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🚀 RUN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 OASIS EVSE DASHBOARD v3.2 SIMPLIFICADO")
    print("="*60)
    print(f"📊 Dataset: {len(DF_FULL):,} registros")
    print(f"🌐 URL: http://127.0.0.1:8050")
    print("="*60 + "\n")
    
    app.run_server(debug=True, port=8050, host="127.0.0.1")