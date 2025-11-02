"""
Gráfico 9: Comparación T1 vs T2
Análisis: Comparación detallada entre tipos de conectores (Tipo 1 vs Tipo 2)
Tipo: Dashboard comparativo multi-métrica
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

from utils import cargar_datos, guardar_grafico

def clasificar_conector(nombre):
    """
    Clasifica el tipo de conector basado en el nombre de la estación
    
    Args:
        nombre: String con el nombre de la estación
    
    Returns:
        String con el tipo de conector (T1, T2, CCS, Otro)
    """
    nombre_lower = nombre.lower()
    if '_t1' in nombre_lower or 't101' in nombre_lower:
        return 'T1'
    elif '_t2' in nombre_lower or 't201' in nombre_lower or 't202' in nombre_lower:
        return 'T2'
    elif 'ccs' in nombre_lower:
        return 'CCS'
    else:
        return 'Otro'

def preparar_datos_comparacion(df):
    """
    Prepara datos agregados por tipo de conector
    
    Args:
        df: DataFrame con datos de Oasis
    
    Returns:
        DataFrame con métricas por tipo de conector
    """
    
    # Clasificar cada transacción
    df['tipo_conector'] = df['evse_uid'].apply(clasificar_conector)
    
    # Agregar por tipo de conector
    metricas_tipo = df.groupby('tipo_conector').agg({
        'id': 'count',  # Transacciones
        'user_id': 'nunique',  # Usuarios únicos
        'evse_uid': 'nunique',  # Estaciones únicas
        'amount_transaction': ['sum', 'mean', 'median'],  # Ingresos
        'energy_kwh': ['sum', 'mean']  # Energía
    }).reset_index()
    
    # Aplanar columnas
    metricas_tipo.columns = ['tipo_conector', 'total_transacciones', 'usuarios_unicos', 
                             'estaciones', 'ingresos_totales', 'ingreso_promedio', 
                             'ingreso_mediano', 'energia_total_kwh', 'energia_promedio_kwh']
    
    # Calcular métricas adicionales
    metricas_tipo['trans_por_estacion'] = metricas_tipo['total_transacciones'] / metricas_tipo['estaciones']
    metricas_tipo['ingresos_por_estacion'] = metricas_tipo['ingresos_totales'] / metricas_tipo['estaciones']
    metricas_tipo['usuarios_por_estacion'] = metricas_tipo['usuarios_unicos'] / metricas_tipo['estaciones']
    metricas_tipo['ingresos_por_usuario'] = metricas_tipo['ingresos_totales'] / metricas_tipo['usuarios_unicos']
    
    return metricas_tipo, df

def preparar_datos_por_estacion(df):
    """
    Prepara datos por estación individual para comparar T1 vs T2
    
    Args:
        df: DataFrame con datos
    
    Returns:
        DataFrame con métricas por estación
    """
    
    estaciones = df.groupby(['evse_uid', 'tipo_conector']).agg({
        'id': 'count',
        'user_id': 'nunique',
        'amount_transaction': 'sum',
        'energy_kwh': 'sum'
    }).reset_index()
    
    estaciones.columns = ['evse_uid', 'tipo_conector', 'transacciones', 
                          'usuarios', 'ingresos', 'energia_kwh']
    
    return estaciones

def crear_grafico_principal(metricas_tipo):
    """
    Crea dashboard comparativo T1 vs T2
    
    Args:
        metricas_tipo: DataFrame con métricas por tipo
    
    Returns:
        Figure de Plotly
    """
    
    # Filtrar solo T1 y T2
    t1_t2 = metricas_tipo[metricas_tipo['tipo_conector'].isin(['T1', 'T2'])]
    
    # Colores
    colors = {'T1': '#3b82f6', 'T2': '#10b981'}
    
    # Crear subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Transacciones Totales',
            'Ingresos Totales',
            'Usuarios Únicos',
            'Energía Consumida (kWh)',
            'Transacciones por Estación',
            'Ingresos por Estación'
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "bar"}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.15
    )
    
    # 1. Transacciones Totales
    fig.add_trace(
        go.Bar(
            x=t1_t2['tipo_conector'],
            y=t1_t2['total_transacciones'],
            marker_color=[colors[t] for t in t1_t2['tipo_conector']],
            text=t1_t2['total_transacciones'],
            textposition='outside',
            name='Transacciones',
            showlegend=False,
            hovertemplate='<b>%{x}</b><br>Transacciones: %{y:,}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 2. Ingresos Totales
    fig.add_trace(
        go.Bar(
            x=t1_t2['tipo_conector'],
            y=t1_t2['ingresos_totales'],
            marker_color=[colors[t] for t in t1_t2['tipo_conector']],
            text=[f"${val/1e6:.1f}M" for val in t1_t2['ingresos_totales']],
            textposition='outside',
            name='Ingresos',
            showlegend=False,
            hovertemplate='<b>%{x}</b><br>Ingresos: $%{y:,.0f} COP<extra></extra>'
        ),
        row=1, col=2
    )
    
    # 3. Usuarios Únicos
    fig.add_trace(
        go.Bar(
            x=t1_t2['tipo_conector'],
            y=t1_t2['usuarios_unicos'],
            marker_color=[colors[t] for t in t1_t2['tipo_conector']],
            text=t1_t2['usuarios_unicos'],
            textposition='outside',
            name='Usuarios',
            showlegend=False,
            hovertemplate='<b>%{x}</b><br>Usuarios: %{y:,}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 4. Energía
    fig.add_trace(
        go.Bar(
            x=t1_t2['tipo_conector'],
            y=t1_t2['energia_total_kwh'],
            marker_color=[colors[t] for t in t1_t2['tipo_conector']],
            text=[f"{val:,.0f} kWh" for val in t1_t2['energia_total_kwh']],
            textposition='outside',
            name='Energía',
            showlegend=False,
            hovertemplate='<b>%{x}</b><br>Energía: %{y:,.0f} kWh<extra></extra>'
        ),
        row=2, col=2
    )
    
    # 5. Transacciones por Estación
    fig.add_trace(
        go.Bar(
            x=t1_t2['tipo_conector'],
            y=t1_t2['trans_por_estacion'],
            marker_color=[colors[t] for t in t1_t2['tipo_conector']],
            text=[f"{val:.0f}" for val in t1_t2['trans_por_estacion']],
            textposition='outside',
            name='Trans/Estación',
            showlegend=False,
            hovertemplate='<b>%{x}</b><br>Trans/Estación: %{y:.1f}<extra></extra>'
        ),
        row=3, col=1
    )
    
    # 6. Ingresos por Estación
    fig.add_trace(
        go.Bar(
            x=t1_t2['tipo_conector'],
            y=t1_t2['ingresos_por_estacion'],
            marker_color=[colors[t] for t in t1_t2['tipo_conector']],
            text=[f"${val/1e6:.1f}M" for val in t1_t2['ingresos_por_estacion']],
            textposition='outside',
            name='$/Estación',
            showlegend=False,
            hovertemplate='<b>%{x}</b><br>Ingresos/Estación: $%{y:,.0f} COP<extra></extra>'
        ),
        row=3, col=2
    )
    
    # Layout
    fig.update_layout(
        title={
            'text': 'Comparación de Conectores: Tipo 1 (T1) vs Tipo 2 (T2)',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 22, 'color': '#2d3748', 'family': 'Arial Black'}
        },
        showlegend=False,
        height=1200,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Arial', size=12)
    )
    
    # Actualizar ejes
    for i in range(1, 4):
        for j in range(1, 3):
            fig.update_xaxes(gridcolor='#e2e8f0', row=i, col=j)
            fig.update_yaxes(gridcolor='#e2e8f0', row=i, col=j)
    
    return fig

# Función de distribución eliminada por solicitud del usuario

def crear_grafico_scatter(estaciones_df):
    """
    Crea scatter plot de transacciones vs ingresos
    
    Args:
        estaciones_df: DataFrame con datos por estación
    
    Returns:
        Figure de Plotly
    """
    
    # Filtrar T1 y T2
    t1_t2 = estaciones_df[estaciones_df['tipo_conector'].isin(['T1', 'T2'])]
    
    fig = px.scatter(
        t1_t2,
        x='transacciones',
        y='ingresos',
        color='tipo_conector',
        size='usuarios',
        color_discrete_map={'T1': '#3b82f6', 'T2': '#10b981'},
        hover_data=['evse_uid', 'energia_kwh'],
        labels={
            'transacciones': 'Transacciones',
            'ingresos': 'Ingresos (COP)',
            'tipo_conector': 'Tipo',
            'usuarios': 'Usuarios'
        },
        title='Transacciones vs Ingresos por Estación (tamaño = usuarios)'
    )
    
    fig.update_layout(
        title={
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#2d3748', 'family': 'Arial Black'}
        },
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=600,
        font=dict(family='Arial', size=12)
    )
    
    fig.update_xaxes(gridcolor='#e2e8f0')
    fig.update_yaxes(gridcolor='#e2e8f0')
    
    return fig

def analizar_t1_vs_t2(metricas_tipo, estaciones_df, df):
    """
    Genera análisis comparativo T1 vs T2
    
    Args:
        metricas_tipo: DataFrame con métricas agregadas
        estaciones_df: DataFrame con datos por estación
        df: DataFrame original
    
    Returns:
        dict con insights
    """
    
    # Filtrar T1 y T2
    t1 = metricas_tipo[metricas_tipo['tipo_conector'] == 'T1'].iloc[0]
    t2 = metricas_tipo[metricas_tipo['tipo_conector'] == 'T2'].iloc[0]
    
    # Calcular diferencias porcentuales
    diff_trans_pct = ((t2['total_transacciones'] - t1['total_transacciones']) / t1['total_transacciones'] * 100)
    diff_ingresos_pct = ((t2['ingresos_totales'] - t1['ingresos_totales']) / t1['ingresos_totales'] * 100)
    diff_usuarios_pct = ((t2['usuarios_unicos'] - t1['usuarios_unicos']) / t1['usuarios_unicos'] * 100)
    
    # Estadísticas por estación
    t1_estaciones = estaciones_df[estaciones_df['tipo_conector'] == 'T1']
    t2_estaciones = estaciones_df[estaciones_df['tipo_conector'] == 'T2']
    
    # Mejor estación de cada tipo
    mejor_t1 = t1_estaciones.nlargest(1, 'transacciones').iloc[0]
    mejor_t2 = t2_estaciones.nlargest(1, 'transacciones').iloc[0]
    
    insights = {
        # T1
        't1_transacciones': int(t1['total_transacciones']),
        't1_ingresos': int(t1['ingresos_totales']),
        't1_usuarios': int(t1['usuarios_unicos']),
        't1_estaciones': int(t1['estaciones']),
        't1_energia': int(t1['energia_total_kwh']),
        't1_trans_por_estacion': t1['trans_por_estacion'],
        't1_ingresos_por_estacion': t1['ingresos_por_estacion'],
        't1_ingreso_promedio': t1['ingreso_promedio'],
        
        # T2
        't2_transacciones': int(t2['total_transacciones']),
        't2_ingresos': int(t2['ingresos_totales']),
        't2_usuarios': int(t2['usuarios_unicos']),
        't2_estaciones': int(t2['estaciones']),
        't2_energia': int(t2['energia_total_kwh']),
        't2_trans_por_estacion': t2['trans_por_estacion'],
        't2_ingresos_por_estacion': t2['ingresos_por_estacion'],
        't2_ingreso_promedio': t2['ingreso_promedio'],
        
        # Diferencias
        'diff_trans_pct': diff_trans_pct,
        'diff_ingresos_pct': diff_ingresos_pct,
        'diff_usuarios_pct': diff_usuarios_pct,
        
        # Ganador
        'ganador_trans': 'T2' if t2['total_transacciones'] > t1['total_transacciones'] else 'T1',
        'ganador_ingresos': 'T2' if t2['ingresos_totales'] > t1['ingresos_totales'] else 'T1',
        'ganador_eficiencia': 'T2' if t2['trans_por_estacion'] > t1['trans_por_estacion'] else 'T1',
        
        # Mejores estaciones
        'mejor_t1_nombre': mejor_t1['evse_uid'],
        'mejor_t1_trans': int(mejor_t1['transacciones']),
        'mejor_t2_nombre': mejor_t2['evse_uid'],
        'mejor_t2_trans': int(mejor_t2['transacciones']),
        
        # Insights
        'insight_volumen': f"T2 supera a T1 en {diff_trans_pct:.1f}% en transacciones totales ({int(t2['total_transacciones']):,} vs {int(t1['total_transacciones']):,}).",
        
        'insight_ingresos': f"T2 genera {diff_ingresos_pct:.1f}% más ingresos que T1 (${int(t2['ingresos_totales']):,} vs ${int(t1['ingresos_totales']):,} COP).",
        
        'insight_eficiencia': f"T2 es más eficiente con {t2['trans_por_estacion']:.0f} trans/estación vs {t1['trans_por_estacion']:.0f} de T1.",
        
        'insight_usuarios': f"T2 atrae {diff_usuarios_pct:.1f}% más usuarios únicos ({int(t2['usuarios_unicos'])} vs {int(t1['usuarios_unicos'])})."
    }
    
    return insights

if __name__ == "__main__":
    print("\n" + "="*80)
    print("GRÁFICO 9: COMPARACIÓN T1 VS T2")
    print("="*80)
    
    # Cargar datos
    csv_path = 'data/df_oasis_clean.csv'
    df = cargar_datos(csv_path)
    print(f"✓ Datos cargados: {len(df):,} registros\n")
    
    # Preparar datos
    print(" Clasificando conectores y preparando métricas...")
    metricas_tipo, df_clasificado = preparar_datos_comparacion(df)
    estaciones_df = preparar_datos_por_estacion(df_clasificado)
    print("✓ Datos preparados\n")
    
    # Crear gráficos
    print("Creando gráfico principal...")
    fig_main = crear_grafico_principal(metricas_tipo)
    
    print(" Creando scatter plot...")
    fig_scatter = crear_grafico_scatter(estaciones_df)
    
    # Analizar
    print("\n" + "="*80)
    print(" ANÁLISIS COMPARATIVO T1 VS T2")
    print("="*80)
    
    insights = analizar_t1_vs_t2(metricas_tipo, estaciones_df, df_clasificado)
    
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
    print(f"   • Transacciones: T2 supera a T1 en {insights['diff_trans_pct']:+.1f}%")
    print(f"   • Ingresos: T2 supera a T1 en {insights['diff_ingresos_pct']:+.1f}%")
    print(f"   • Usuarios: T2 supera a T1 en {insights['diff_usuarios_pct']:+.1f}%")
    
    print(f"\n GANADORES POR CATEGORÍA:")
    print(f"   • Volumen (transacciones): {insights['ganador_trans']}")
    print(f"   • Ingresos totales: {insights['ganador_ingresos']}")
    print(f"   • Eficiencia (trans/estación): {insights['ganador_eficiencia']}")
    
    print(f"\n MEJORES ESTACIONES:")
    print(f"   • Mejor T1: {insights['mejor_t1_nombre']} ({insights['mejor_t1_trans']:,} trans)")
    print(f"   • Mejor T2: {insights['mejor_t2_nombre']} ({insights['mejor_t2_trans']:,} trans)")
    
    print(f"\n INSIGHTS PRINCIPALES:")
    print(f"\n   1. VOLUMEN:")
    print(f"      {insights['insight_volumen']}")
    
    print(f"\n   2. INGRESOS:")
    print(f"      {insights['insight_ingresos']}")
    
    print(f"\n   3. EFICIENCIA:")
    print(f"      {insights['insight_eficiencia']}")
    
    print(f"\n   4. USUARIOS:")
    print(f"      {insights['insight_usuarios']}")
    
    print(f"\n   5. RECOMENDACIÓN ESTRATÉGICA:")
    if insights['ganador_trans'] == 'T2' and insights['ganador_ingresos'] == 'T2':
        print(f"      T2 es superior en todas las métricas clave.")
        print(f"      Recomendación: Priorizar instalación de conectores T2 en nuevas estaciones.")
    else:
        print(f"      Los resultados son mixtos. Analizar caso por caso según ubicación.")
    
    # Guardar gráficos
    print("\n" + "="*80)
    print(" Guardando gráficos...")
    guardar_grafico(fig_main, 'grafico_09_comparacion_t1_t2_principal.png')
    guardar_grafico(fig_scatter, 'grafico_09_scatter.png')
    
    # Mostrar en navegador
    print(" Abriendo gráfico principal en navegador...")
    fig_main.show()
    
    print("\n Abriendo scatter plot en navegador...")
    fig_scatter.show()
    
    