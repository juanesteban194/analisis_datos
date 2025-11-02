"""
Gráfico 8: Top Estaciones - Análisis Comparativo Detallado
Análisis: Ranking de estaciones con métricas múltiples y comparación de rendimiento
Tipo: Dashboard multi-gráfico con métricas de rendimiento
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
import os
import sys

# Agregar el directorio raíz al path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from utils import cargar_datos, guardar_grafico, formatear_moneda, formatear_numero

def calcular_metricas_estaciones(df):
    """
    Calcula métricas completas por estación
    
    Args:
        df: DataFrame con datos de Oasis
    
    Returns:
        DataFrame con métricas por estación
    """
    
    # Agregar por estación
    metricas = df.groupby('evse_uid').agg({
        'id': 'count',  # Total de transacciones
        'user_id': 'nunique',  # Usuarios únicos
        'amount_transaction': ['sum', 'mean', 'median'],  # Ingresos
        'energy_kwh': ['sum', 'mean'],  # Energía
        'start_date_time': ['min', 'max']  # Rango de fechas
    }).reset_index()
    
    # Aplanar columnas multi-nivel
    metricas.columns = ['evse_uid', 'total_transacciones', 'usuarios_unicos',
                        'ingresos_totales', 'ingreso_promedio', 'ingreso_mediano',
                        'energia_total_kwh', 'energia_promedio_kwh',
                        'primera_fecha', 'ultima_fecha']
    
    # Calcular días activos
    metricas['dias_activos'] = (metricas['ultima_fecha'] - metricas['primera_fecha']).dt.days + 1
    
    # Métricas de eficiencia
    metricas['transacciones_por_dia'] = metricas['total_transacciones'] / metricas['dias_activos']
    metricas['ingresos_por_dia'] = metricas['ingresos_totales'] / metricas['dias_activos']
    metricas['usuarios_por_dia'] = metricas['usuarios_unicos'] / metricas['dias_activos']
    
    # Índice de eficiencia combinado (normalizado)
    # Combina: transacciones, ingresos y usuarios
    norm_trans = (metricas['total_transacciones'] - metricas['total_transacciones'].min()) / (metricas['total_transacciones'].max() - metricas['total_transacciones'].min())
    norm_ingresos = (metricas['ingresos_totales'] - metricas['ingresos_totales'].min()) / (metricas['ingresos_totales'].max() - metricas['ingresos_totales'].min())
    norm_usuarios = (metricas['usuarios_unicos'] - metricas['usuarios_unicos'].min()) / (metricas['usuarios_unicos'].max() - metricas['usuarios_unicos'].min())
    
    metricas['indice_rendimiento'] = (norm_trans * 0.4 + norm_ingresos * 0.4 + norm_usuarios * 0.2) * 100
    
    # Categorización de estaciones
    def categorizar_estacion(row):
        if row['indice_rendimiento'] >= 80:
            return 'Elite'
        elif row['indice_rendimiento'] >= 60:
            return 'Alto Rendimiento'
        elif row['indice_rendimiento'] >= 40:
            return 'Rendimiento Medio'
        elif row['indice_rendimiento'] >= 20:
            return 'Bajo Rendimiento'
        else:
            return 'Crítico'
    
    metricas['categoria'] = metricas.apply(categorizar_estacion, axis=1)
    
    # Identificar tipo de conector
    def identificar_tipo(nombre):
        nombre_lower = nombre.lower()
        if 'ccs' in nombre_lower:
            return 'CCS'
        elif 't1' in nombre_lower:
            return 'Tipo 1'
        elif 't2' in nombre_lower:
            return 'Tipo 2'
        else:
            return 'Otro'
    
    metricas['tipo_conector'] = metricas['evse_uid'].apply(identificar_tipo)
    
    # Identificar ubicación (si es Éxito)
    metricas['es_exito'] = metricas['evse_uid'].str.lower().str.contains('exito', na=False)
    
    # Ordenar por índice de rendimiento
    metricas = metricas.sort_values('indice_rendimiento', ascending=False)
    
    return metricas

def crear_grafico_principal(metricas_df, top_n=15):
    """
    Crea dashboard con top estaciones
    
    Args:
        metricas_df: DataFrame con métricas
        top_n: Número de estaciones top a mostrar
    
    Returns:
        Figure de Plotly
    """
    
    # Tomar top N estaciones
    top_estaciones = metricas_df.head(top_n)
    
    # Colores por categoría
    color_map = {
        'Elite': '#10b981',
        'Alto Rendimiento': '#3b82f6',
        'Rendimiento Medio': '#f59e0b',
        'Bajo Rendimiento': '#ef4444',
        'Crítico': '#dc2626'
    }
    
    colors = [color_map[cat] for cat in top_estaciones['categoria']]
    
    # Crear subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            f'Top {top_n} Estaciones por Índice de Rendimiento',
            f'Top {top_n} por Ingresos Totales',
            f'Top {top_n} por Transacciones',
            f'Top {top_n} por Usuarios Únicos',
            'Eficiencia: Transacciones por Día',
            'Eficiencia: Ingresos por Día'
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "bar"}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.15,
        row_heights=[0.35, 0.35, 0.30]
    )
    
    # 1. Índice de Rendimiento
    fig.add_trace(
        go.Bar(
            y=top_estaciones['evse_uid'],
            x=top_estaciones['indice_rendimiento'],
            orientation='h',
            marker_color=colors,
            text=[f"{val:.1f}" for val in top_estaciones['indice_rendimiento']],
            textposition='outside',
            name='Índice',
            hovertemplate='<b>%{y}</b><br>Índice: %{x:.1f}<br>Categoría: ' + top_estaciones['categoria'] + '<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 2. Ingresos Totales
    top_ingresos = metricas_df.nlargest(top_n, 'ingresos_totales')
    fig.add_trace(
        go.Bar(
            y=top_ingresos['evse_uid'],
            x=top_ingresos['ingresos_totales'],
            orientation='h',
            marker_color='#10b981',
            text=[f"${val/1e6:.1f}M" for val in top_ingresos['ingresos_totales']],
            textposition='outside',
            name='Ingresos',
            hovertemplate='<b>%{y}</b><br>Ingresos: $%{x:,.0f} COP<extra></extra>'
        ),
        row=1, col=2
    )
    
    # 3. Transacciones
    top_trans = metricas_df.nlargest(top_n, 'total_transacciones')
    fig.add_trace(
        go.Bar(
            y=top_trans['evse_uid'],
            x=top_trans['total_transacciones'],
            orientation='h',
            marker_color='#3b82f6',
            text=top_trans['total_transacciones'].astype(str),
            textposition='outside',
            name='Transacciones',
            hovertemplate='<b>%{y}</b><br>Transacciones: %{x}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 4. Usuarios Únicos
    top_usuarios = metricas_df.nlargest(top_n, 'usuarios_unicos')
    fig.add_trace(
        go.Bar(
            y=top_usuarios['evse_uid'],
            x=top_usuarios['usuarios_unicos'],
            orientation='h',
            marker_color='#8b5cf6',
            text=top_usuarios['usuarios_unicos'].astype(str),
            textposition='outside',
            name='Usuarios',
            hovertemplate='<b>%{y}</b><br>Usuarios: %{x}<extra></extra>'
        ),
        row=2, col=2
    )
    
    # 5. Eficiencia - Trans/Día
    top_efic_trans = metricas_df.nlargest(top_n, 'transacciones_por_dia')
    fig.add_trace(
        go.Bar(
            y=top_efic_trans['evse_uid'],
            x=top_efic_trans['transacciones_por_dia'],
            orientation='h',
            marker_color='#f59e0b',
            text=[f"{val:.1f}" for val in top_efic_trans['transacciones_por_dia']],
            textposition='outside',
            name='Trans/Día',
            hovertemplate='<b>%{y}</b><br>Trans/Día: %{x:.2f}<extra></extra>'
        ),
        row=3, col=1
    )
    
    # 6. Eficiencia - Ingresos/Día
    top_efic_ingresos = metricas_df.nlargest(top_n, 'ingresos_por_dia')
    fig.add_trace(
        go.Bar(
            y=top_efic_ingresos['evse_uid'],
            x=top_efic_ingresos['ingresos_por_dia'],
            orientation='h',
            marker_color='#ec4899',
            text=[f"${val/1e3:.0f}K" for val in top_efic_ingresos['ingresos_por_dia']],
            textposition='outside',
            name='$/Día',
            hovertemplate='<b>%{y}</b><br>Ingresos/Día: $%{x:,.0f} COP<extra></extra>'
        ),
        row=3, col=2
    )
    
    # Layout
    fig.update_layout(
        title={
            'text': 'Análisis Comparativo: Top Estaciones de Carga',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 22, 'color': '#2d3748', 'family': 'Arial Black'}
        },
        showlegend=False,
        height=1400,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Arial', size=10)
    )
    
    # Actualizar ejes
    for i in range(1, 4):
        for j in range(1, 3):
            fig.update_xaxes(gridcolor='#e2e8f0', row=i, col=j)
            fig.update_yaxes(tickfont={'size': 8}, row=i, col=j)
    
    # Títulos de ejes X
    fig.update_xaxes(title_text="Índice de Rendimiento (0-100)", row=1, col=1)
    fig.update_xaxes(title_text="Ingresos Totales (COP)", row=1, col=2)
    fig.update_xaxes(title_text="Número de Transacciones", row=2, col=1)
    fig.update_xaxes(title_text="Usuarios Únicos", row=2, col=2)
    fig.update_xaxes(title_text="Transacciones por Día", row=3, col=1)
    fig.update_xaxes(title_text="Ingresos por Día (COP)", row=3, col=2)
    
    return fig

def crear_grafico_radar(metricas_df, top_n=8):
    """
    Crea gráfico de radar comparando top estaciones
    
    Args:
        metricas_df: DataFrame con métricas
        top_n: Número de estaciones a comparar
    
    Returns:
        Figure de Plotly
    """
    
    # Tomar top N
    top = metricas_df.head(top_n)
    
    # Normalizar métricas para el radar (0-100)
    def normalize(series):
        return (series - series.min()) / (series.max() - series.min()) * 100
    
    # Crear figura
    fig = go.Figure()
    
    # Métricas a comparar
    metricas_radar = ['total_transacciones', 'usuarios_unicos', 'ingresos_totales', 
                      'energia_total_kwh', 'transacciones_por_dia']
    labels_radar = ['Transacciones', 'Usuarios', 'Ingresos', 'Energía', 'Eficiencia']
    
    # Colores
    colors = px.colors.qualitative.Set2[:top_n]
    
    # Añadir cada estación
    for idx, (_, row) in enumerate(top.iterrows()):
        valores = [
            normalize(metricas_df['total_transacciones'])[row.name],
            normalize(metricas_df['usuarios_unicos'])[row.name],
            normalize(metricas_df['ingresos_totales'])[row.name],
            normalize(metricas_df['energia_total_kwh'])[row.name],
            normalize(metricas_df['transacciones_por_dia'])[row.name]
        ]
        
        fig.add_trace(go.Scatterpolar(
            r=valores,
            theta=labels_radar,
            fill='toself',
            name=row['evse_uid'][:20] + '...' if len(row['evse_uid']) > 20 else row['evse_uid'],
            line_color=colors[idx],
            opacity=0.7
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor='#e2e8f0'
            )
        ),
        title={
            'text': f'Comparación Multi-Dimensional: Top {top_n} Estaciones',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#2d3748', 'family': 'Arial Black'}
        },
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.1
        ),
        height=700,
        font=dict(family='Arial', size=11)
    )
    
    return fig

def crear_grafico_categorias(metricas_df):
    """
    Crea gráfico de distribución por categorías
    
    Args:
        metricas_df: DataFrame con métricas
    
    Returns:
        Figure de Plotly
    """
    
    # Contar por categoría
    categoria_counts = metricas_df['categoria'].value_counts()
    categoria_ingresos = metricas_df.groupby('categoria')['ingresos_totales'].sum()
    
    # Colores por categoría
    color_map = {
        'Elite': '#10b981',
        'Alto Rendimiento': '#3b82f6',
        'Rendimiento Medio': '#f59e0b',
        'Bajo Rendimiento': '#ef4444',
        'Crítico': '#dc2626'
    }
    
    # Orden de categorías
    orden = ['Elite', 'Alto Rendimiento', 'Rendimiento Medio', 'Bajo Rendimiento', 'Crítico']
    categoria_counts = categoria_counts.reindex(orden, fill_value=0)
    categoria_ingresos = categoria_ingresos.reindex(orden, fill_value=0)
    
    # Crear subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Distribución de Estaciones por Categoría', 
                       'Ingresos por Categoría de Estación'),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Gráfico 1: Cantidad de estaciones
    colors_cat = [color_map[cat] for cat in categoria_counts.index]
    fig.add_trace(
        go.Bar(
            x=categoria_counts.index,
            y=categoria_counts.values,
            marker_color=colors_cat,
            text=categoria_counts.values,
            textposition='outside',
            name='Estaciones',
            hovertemplate='<b>%{x}</b><br>Estaciones: %{y}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Gráfico 2: Ingresos
    fig.add_trace(
        go.Bar(
            x=categoria_ingresos.index,
            y=categoria_ingresos.values,
            marker_color=colors_cat,
            text=[f"${val/1e6:.1f}M" for val in categoria_ingresos.values],
            textposition='outside',
            name='Ingresos',
            hovertemplate='<b>%{x}</b><br>Ingresos: $%{y:,.0f} COP<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Layout
    fig.update_layout(
        title={
            'text': 'Análisis por Categoría de Rendimiento',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#2d3748', 'family': 'Arial Black'}
        },
        showlegend=False,
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Arial', size=12)
    )
    
    fig.update_xaxes(tickangle=-15, gridcolor='#e2e8f0')
    fig.update_yaxes(gridcolor='#e2e8f0')
    
    fig.update_yaxes(title_text="Número de Estaciones", row=1, col=1)
    fig.update_yaxes(title_text="Ingresos Totales (COP)", row=1, col=2)
    
    return fig

def analizar_top_estaciones(metricas_df, df):
    """
    Genera análisis e insights de top estaciones
    
    Args:
        metricas_df: DataFrame con métricas
        df: DataFrame original
    
    Returns:
        dict con insights
    """
    
    # Top 10 estaciones
    top_10 = metricas_df.head(10)
    
    # Estadísticas generales
    total_estaciones = len(metricas_df)
    
    # Top 1
    top_1 = metricas_df.iloc[0]
    
    # Distribución por categoría
    categoria_dist = metricas_df['categoria'].value_counts()
    
    # Comparación Éxito vs No Éxito
    exito_stats = metricas_df[metricas_df['es_exito']].agg({
        'total_transacciones': 'sum',
        'ingresos_totales': 'sum',
        'usuarios_unicos': 'sum'
    })
    
    no_exito_stats = metricas_df[~metricas_df['es_exito']].agg({
        'total_transacciones': 'sum',
        'ingresos_totales': 'sum',
        'usuarios_unicos': 'sum'
    })
    
    # Concentración en top 10
    total_trans = metricas_df['total_transacciones'].sum()
    total_ingresos = metricas_df['ingresos_totales'].sum()
    
    concentracion_trans_top10 = top_10['total_transacciones'].sum() / total_trans * 100
    concentracion_ingresos_top10 = top_10['ingresos_totales'].sum() / total_ingresos * 100
    
    # Rango de rendimiento
    mejor_indice = metricas_df['indice_rendimiento'].max()
    peor_indice = metricas_df['indice_rendimiento'].min()
    promedio_indice = metricas_df['indice_rendimiento'].mean()
    
    insights = {
        'total_estaciones': total_estaciones,
        
        'top_1_nombre': top_1['evse_uid'],
        'top_1_indice': top_1['indice_rendimiento'],
        'top_1_trans': int(top_1['total_transacciones']),
        'top_1_ingresos': int(top_1['ingresos_totales']),
        'top_1_categoria': top_1['categoria'],
        
        'concentracion_trans_top10': concentracion_trans_top10,
        'concentracion_ingresos_top10': concentracion_ingresos_top10,
        
        'categoria_elite': int(categoria_dist.get('Elite', 0)),
        'categoria_alto': int(categoria_dist.get('Alto Rendimiento', 0)),
        'categoria_medio': int(categoria_dist.get('Rendimiento Medio', 0)),
        'categoria_bajo': int(categoria_dist.get('Bajo Rendimiento', 0)),
        'categoria_critico': int(categoria_dist.get('Crítico', 0)),
        
        'mejor_indice': mejor_indice,
        'peor_indice': peor_indice,
        'promedio_indice': promedio_indice,
        
        'exito_trans': int(exito_stats['total_transacciones']),
        'exito_ingresos': int(exito_stats['ingresos_totales']),
        'exito_count': int(metricas_df['es_exito'].sum()),
        
        'no_exito_trans': int(no_exito_stats['total_transacciones']),
        'no_exito_ingresos': int(no_exito_stats['ingresos_totales']),
        'no_exito_count': int((~metricas_df['es_exito']).sum()),
        
        'top_10': top_10,
        
        'insight_lider': f"La estación líder es '{top_1['evse_uid']}' (categoría {top_1['categoria']}) con un índice de rendimiento de {top_1['indice_rendimiento']:.1f}/100, {int(top_1['total_transacciones'])} transacciones y ${int(top_1['ingresos_totales']):,} COP en ingresos.",
        
        'insight_concentracion': f"Las top 10 estaciones concentran el {concentracion_trans_top10:.1f}% de las transacciones y el {concentracion_ingresos_top10:.1f}% de los ingresos totales, demostrando alta concentración.",
        
        'insight_distribucion': f"De {total_estaciones} estaciones: {int(categoria_dist.get('Elite', 0))} son Elite, {int(categoria_dist.get('Alto Rendimiento', 0))} de Alto Rendimiento, {int(categoria_dist.get('Rendimiento Medio', 0))} de Rendimiento Medio, {int(categoria_dist.get('Bajo Rendimiento', 0))} de Bajo Rendimiento y {int(categoria_dist.get('Crítico', 0))} en estado Crítico.",
        
        'insight_exito': f"Las estaciones Éxito ({int(metricas_df['es_exito'].sum())} estaciones) generan {int(exito_stats['total_transacciones'])} transacciones (${int(exito_stats['ingresos_totales']):,} COP) vs {int(no_exito_stats['total_transacciones'])} transacciones de otras ubicaciones."
    }
    
    return insights

if __name__ == "__main__":
    print("\n" + "="*80)
    print("GRÁFICO 8: TOP ESTACIONES - ANÁLISIS COMPARATIVO")
    print("="*80)
    
    # Cargar datos
    csv_path = 'data/df_oasis_clean.csv'
    df = cargar_datos(csv_path)
    print(f"✓ Datos cargados: {len(df):,} registros\n")
    
    # Calcular métricas
    print(" Calculando métricas por estación...")
    metricas_df = calcular_metricas_estaciones(df)
    print(f"✓ Métricas calculadas para {len(metricas_df)} estaciones\n")
    
    # Crear gráficos
    print(" Creando gráfico principal...")
    fig_main = crear_grafico_principal(metricas_df, top_n=15)
    
    print("  Creando gráfico de radar...")
    fig_radar = crear_grafico_radar(metricas_df, top_n=8)
    
    print(" Creando análisis por categorías...")
    fig_categorias = crear_grafico_categorias(metricas_df)
    
    # Analizar
    print("\n" + "="*80)
    print(" ANÁLISIS DE TOP ESTACIONES")
    print("="*80)
    
    insights = analizar_top_estaciones(metricas_df, df)
    
    print(f"\n RESUMEN GENERAL:")
    print(f"   • Total de estaciones: {insights['total_estaciones']}")
    print(f"   • Rango de índice: {insights['peor_indice']:.1f} - {insights['mejor_indice']:.1f}")
    print(f"   • Índice promedio: {insights['promedio_indice']:.1f}")
    
    print(f"\n ESTACIÓN #1:")
    print(f"   • Nombre: {insights['top_1_nombre']}")
    print(f"   • Categoría: {insights['top_1_categoria']}")
    print(f"   • Índice de rendimiento: {insights['top_1_indice']:.1f}/100")
    print(f"   • Transacciones: {insights['top_1_trans']:,}")
    print(f"   • Ingresos: ${insights['top_1_ingresos']:,} COP")
    
    print(f"\n CONCENTRACIÓN TOP 10:")
    print(f"   • % de transacciones: {insights['concentracion_trans_top10']:.1f}%")
    print(f"   • % de ingresos: {insights['concentracion_ingresos_top10']:.1f}%")
    
    print(f"\n DISTRIBUCIÓN POR CATEGORÍA:")
    print(f"   • Elite: {insights['categoria_elite']} estaciones")
    print(f"   • Alto Rendimiento: {insights['categoria_alto']} estaciones")
    print(f"   • Rendimiento Medio: {insights['categoria_medio']} estaciones")
    print(f"   • Bajo Rendimiento: {insights['categoria_bajo']} estaciones")
    print(f"   • Crítico: {insights['categoria_critico']} estaciones")
    
    print(f"\n COMPARACIÓN ÉXITO:")
    print(f"   • Estaciones Éxito: {insights['exito_count']}")
    print(f"   • Transacciones: {insights['exito_trans']:,}")
    print(f"   • Ingresos: ${insights['exito_ingresos']:,} COP")
    print(f"\n   • Otras estaciones: {insights['no_exito_count']}")
    print(f"   • Transacciones: {insights['no_exito_trans']:,}")
    print(f"   • Ingresos: ${insights['no_exito_ingresos']:,} COP")
    
    print(f"\n TOP 10 ESTACIONES:")
    print(f"\n{'#':<3} {'Estación':<30} {'Índice':>8} {'Trans':>8} {'Usuarios':>10} {'Ingresos':>15} {'Categoría':<20}")
    print("-" * 110)
    for idx, (_, row) in enumerate(insights['top_10'].iterrows(), 1):
        print(f"{idx:<3} {row['evse_uid'][:30]:<30} {row['indice_rendimiento']:>8.1f} {int(row['total_transacciones']):>8} {int(row['usuarios_unicos']):>10} ${int(row['ingresos_totales']):>14,} {row['categoria']:<20}")
    
    print(f"\n INSIGHTS PRINCIPALES:")
    print(f"\n   1. ESTACIÓN LÍDER:")
    print(f"      {insights['insight_lider']}")
    
    print(f"\n   2. CONCENTRACIÓN:")
    print(f"      {insights['insight_concentracion']}")
    
    print(f"\n   3. DISTRIBUCIÓN:")
    print(f"      {insights['insight_distribucion']}")
    
    print(f"\n   4. MODELO ÉXITO:")
    print(f"      {insights['insight_exito']}")
    
    print(f"\n   5. RECOMENDACIONES ESTRATÉGICAS:")
    print(f"      • Priorizar inversión en estaciones Elite y Alto Rendimiento")
    print(f"      • Evaluar estaciones en categoría Crítico para mejora o cierre")
    print(f"      • Replicar mejores prácticas de top performers")
    print(f"      • Expandir modelo Éxito a nuevas ubicaciones")
    print(f"      • Optimizar estaciones de Rendimiento Medio con potencial")
    
    # Guardar gráficos
    print("\n" + "="*80)
    print(" Guardando gráficos...")
    guardar_grafico(fig_main, 'grafico_08_top_estaciones_principal.png')
    guardar_grafico(fig_radar, 'grafico_08_top_estaciones_radar.png')
    guardar_grafico(fig_categorias, 'grafico_08_categorias.png')
    
    # Mostrar en navegador
    print(" Abriendo gráfico principal en navegador...")
    fig_main.show()
    
    print("\n Abriendo gráfico de radar en navegador...")
    fig_radar.show()
    
    print("\n Abriendo análisis por categorías en navegador...")
    fig_categorias.show()
    
   