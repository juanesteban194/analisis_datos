"""
Gráfico 7: Heatmap de Uso de Estaciones
Análisis: Uso detallado por estación con comparación temporal
Tipo: Heatmap de estaciones vs tiempo + análisis comparativo
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

def preparar_datos_uso(df):
    """
    Prepara datos de uso por estación
    
    Args:
        df: DataFrame con datos de Oasis
    
    Returns:
        DataFrame con métricas por estación
    """
    
    # Agregar por estación (usar 'id' en lugar de 'transaction_id')
    uso_por_estacion = df.groupby('evse_uid').agg({
        'id': 'count',  # Número de transacciones
        'user_id': 'nunique',  # Usuarios únicos
        'amount_transaction': ['sum', 'mean'],  # Ingresos
        'energy_kwh': ['sum', 'mean'],  # Energía
        'start_date_time': ['min', 'max']  # Primera y última transacción
    })
    
    # Aplanar columnas
    uso_por_estacion.columns = ['transacciones', 'usuarios_unicos', 
                                 'ingresos_totales', 'ingreso_promedio',
                                 'energia_total', 'energia_promedio',
                                 'primera_transaccion', 'ultima_transaccion']
    
    uso_por_estacion = uso_por_estacion.reset_index()
    
    # Calcular métricas adicionales
    uso_por_estacion['dias_activo'] = (
        uso_por_estacion['ultima_transaccion'] - uso_por_estacion['primera_transaccion']
    ).dt.days + 1
    
    uso_por_estacion['trans_por_dia'] = (
        uso_por_estacion['transacciones'] / uso_por_estacion['dias_activo']
    )
    
    uso_por_estacion['ingresos_por_dia'] = (
        uso_por_estacion['ingresos_totales'] / uso_por_estacion['dias_activo']
    )
    
    # Identificar tipo de estación (si tiene patrón en el nombre)
    def identificar_tipo(nombre):
        nombre_lower = nombre.lower()
        if 'exito' in nombre_lower:
            return 'Éxito'
        elif 'ccs' in nombre_lower:
            return 'CCS'
        elif 't1' in nombre_lower or 't2' in nombre_lower:
            return 'Tipo 1/2'
        else:
            return 'Otro'
    
    uso_por_estacion['tipo'] = uso_por_estacion['evse_uid'].apply(identificar_tipo)
    
    # Ordenar por transacciones
    uso_por_estacion = uso_por_estacion.sort_values('transacciones', ascending=False)
    
    return uso_por_estacion

def crear_matriz_temporal(df):
    """
    Crea matriz de uso por estación a lo largo del tiempo
    
    Args:
        df: DataFrame con datos
    
    Returns:
        DataFrame pivotado
    """
    
    # Crear columna año-mes
    df['year_month'] = df['start_date_time'].dt.to_period('M').astype(str)
    
    # Agregar por estación y mes
    matriz = df.groupby(['evse_uid', 'year_month']).size().reset_index(name='transacciones')
    
    # Pivotar
    matriz_pivot = matriz.pivot(index='evse_uid', columns='year_month', values='transacciones')
    matriz_pivot = matriz_pivot.fillna(0)
    
    # Ordenar por total de transacciones
    matriz_pivot['total'] = matriz_pivot.sum(axis=1)
    matriz_pivot = matriz_pivot.sort_values('total', ascending=False).drop('total', axis=1)
    
    return matriz_pivot

def crear_grafico(uso_df, matriz_temporal):
    """
    Crea visualizaciones de uso por estación
    
    Args:
        uso_df: DataFrame con métricas por estación
        matriz_temporal: Matriz temporal de uso
    
    Returns:
        Figure de Plotly
    """
    
    # Tomar top 20 estaciones para mejor visualización
    top_20 = uso_df.head(20)
    matriz_top = matriz_temporal.head(20)
    
    # Crear subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Heatmap: Uso de Estaciones en el Tiempo (Top 20)',
            'Top 20 Estaciones por Transacciones',
            'Ingresos por Estación (Top 20)',
            'Usuarios Únicos por Estación (Top 20)'
        ),
        specs=[
            [{"type": "heatmap", "colspan": 2}, None],
            [{"type": "bar"}, {"type": "bar"}]
        ],
        row_heights=[0.5, 0.5],
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )
    
    # Heatmap temporal
    fig.add_trace(
        go.Heatmap(
            z=matriz_top.values,
            x=matriz_top.columns,
            y=matriz_top.index,
            colorscale='Viridis',
            hovertemplate='<b>%{y}</b><br>Mes: %{x}<br>Transacciones: %{z}<extra></extra>',
            colorbar=dict(title="Transacciones", x=1.02)
        ),
        row=1, col=1
    )
    
    # Top 20 por transacciones
    colors_trans = px.colors.sequential.Blues_r[:20]
    fig.add_trace(
        go.Bar(
            y=top_20['evse_uid'],
            x=top_20['transacciones'],
            orientation='h',
            marker_color=colors_trans,
            text=top_20['transacciones'],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Transacciones: %{x}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Ingresos por estación
    fig.add_trace(
        go.Bar(
            y=top_20['evse_uid'],
            x=top_20['ingresos_totales'],
            orientation='h',
            marker_color='#10b981',
            text=[f"${val/1e6:.1f}M" for val in top_20['ingresos_totales']],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Ingresos: $%{x:,.0f} COP<extra></extra>'
        ),
        row=2, col=2
    )
    
    # Layout
    fig.update_layout(
        title={
            'text': 'Análisis de Uso por Estación',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2d3748', 'family': 'Arial Black'}
        },
        showlegend=False,
        height=1000,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Arial', size=10)
    )
    
    # Actualizar ejes
    fig.update_xaxes(title_text="Mes", tickangle=-45, row=1, col=1)
    fig.update_yaxes(title_text="Estación", tickfont={'size': 8}, row=1, col=1)
    
    fig.update_xaxes(title_text="Transacciones", gridcolor='#e2e8f0', row=2, col=1)
    fig.update_yaxes(tickfont={'size': 8}, row=2, col=1)
    
    fig.update_xaxes(title_text="Ingresos (COP)", gridcolor='#e2e8f0', row=2, col=2)
    fig.update_yaxes(tickfont={'size': 8}, row=2, col=2)
    
    return fig

def crear_grafico_comparativo(uso_df):
    """
    Crea gráfico de dispersión comparativo
    
    Args:
        uso_df: DataFrame con métricas
    
    Returns:
        Figure de Plotly
    """
    
    # Colores por tipo
    color_map = {
        'Éxito': '#10b981',
        'CCS': '#3b82f6',
        'Tipo 1/2': '#8b5cf6',
        'Otro': '#f59e0b'
    }
    
    fig = px.scatter(
        uso_df,
        x='transacciones',
        y='ingresos_totales',
        size='usuarios_unicos',
        color='tipo',
        color_discrete_map=color_map,
        hover_data=['evse_uid', 'trans_por_dia', 'energia_total'],
        labels={
            'transacciones': 'Número de Transacciones',
            'ingresos_totales': 'Ingresos Totales (COP)',
            'tipo': 'Tipo de Estación',
            'usuarios_unicos': 'Usuarios Únicos'
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

def crear_grafico_eficiencia(uso_df):
    """
    Crea gráfico de eficiencia (transacciones por día)
    
    Args:
        uso_df: DataFrame con métricas
    
    Returns:
        Figure de Plotly
    """
    
    # Top 20 por eficiencia
    top_eficiencia = uso_df.nlargest(20, 'trans_por_dia')
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=top_eficiencia['evse_uid'],
        y=top_eficiencia['trans_por_dia'],
        marker=dict(
            color=top_eficiencia['trans_por_dia'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Trans/Día")
        ),
        text=[f"{val:.1f}" for val in top_eficiencia['trans_por_dia']],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Transacciones/día: %{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': 'Top 20 Estaciones por Eficiencia (Transacciones por Día)',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#2d3748', 'family': 'Arial Black'}
        },
        xaxis_title="Estación",
        yaxis_title="Transacciones por Día",
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=600,
        font=dict(family='Arial', size=11)
    )
    
    fig.update_xaxes(tickangle=-45, gridcolor='#e2e8f0')
    fig.update_yaxes(gridcolor='#e2e8f0')
    
    return fig

def analizar(uso_df, df):
    """
    Analiza uso por estación y genera insights
    
    Args:
        uso_df: DataFrame con métricas por estación
        df: DataFrame original
    
    Returns:
        dict con insights
    """
    
    total_estaciones = len(uso_df)
    
    # Top 3 estaciones
    top_3 = uso_df.head(3)
    
    # Estaciones de bajo rendimiento (<50 trans)
    bajo_rendimiento = uso_df[uso_df['transacciones'] < 50]
    
    # Estadísticas generales
    promedio_trans = uso_df['transacciones'].mean()
    
    
    # Por tipo de estación
    stats_por_tipo = uso_df.groupby('tipo').agg({
        'evse_uid': 'count',
        'transacciones': ['sum', 'mean'],
        'ingresos_totales': ['sum', 'mean']
    }).round(2)
    
    # Concentración (top 3)
    total_trans = uso_df['transacciones'].sum()
    concentracion_top3 = top_3['transacciones'].sum() / total_trans * 100
    
    # Eficiencia
    top_eficiencia = uso_df.nlargest(3, 'trans_por_dia')
    
    insights = {
        'total_estaciones': total_estaciones,
        'total_transacciones': int(total_trans),
        
        'top_1_nombre': top_3.iloc[0]['evse_uid'],
        'top_1_trans': int(top_3.iloc[0]['transacciones']),
        'top_1_ingresos': int(top_3.iloc[0]['ingresos_totales']),
        
        'top_2_nombre': top_3.iloc[1]['evse_uid'],
        'top_2_trans': int(top_3.iloc[1]['transacciones']),
        
        'top_3_nombre': top_3.iloc[2]['evse_uid'],
        'top_3_trans': int(top_3.iloc[2]['transacciones']),
        
        'concentracion_top3': concentracion_top3,
        
        'promedio_trans': promedio_trans,
        
        
        'bajo_rendimiento_count': len(bajo_rendimiento),
        'bajo_rendimiento_pct': len(bajo_rendimiento) / total_estaciones * 100,
        
        'top_eficiencia_nombre': top_eficiencia.iloc[0]['evse_uid'],
        'top_eficiencia_valor': top_eficiencia.iloc[0]['trans_por_dia'],
        
        'stats_por_tipo': stats_por_tipo,
        
        'insight_top': f"La estación líder es '{top_3.iloc[0]['evse_uid']}' con {int(top_3.iloc[0]['transacciones'])} transacciones y ${int(top_3.iloc[0]['ingresos_totales']):,} COP en ingresos.",
        
        'insight_concentracion': f"Las top 3 estaciones ({top_3.iloc[0]['evse_uid']}, {top_3.iloc[1]['evse_uid']}, {top_3.iloc[2]['evse_uid']}) concentran el {concentracion_top3:.1f}% de todas las transacciones.",
        
        'insight_bajo_rendimiento': f"{len(bajo_rendimiento)} estaciones ({len(bajo_rendimiento)/total_estaciones*100:.1f}%) tienen menos de 50 transacciones. Son candidatas para evaluación o reubicación.",
        
        'insight_eficiencia': f"La estación más eficiente es '{top_eficiencia.iloc[0]['evse_uid']}' con {top_eficiencia.iloc[0]['trans_por_dia']:.1f} transacciones por día.",
        
    }
    
    return insights

if __name__ == "__main__":
    print("\n" + "="*80)
    print("GRÁFICO 7: HEATMAP DE USO DE ESTACIONES")
    print("="*80)
    
    # Cargar datos
    csv_path = 'data/df_oasis_clean.csv'
    df = cargar_datos(csv_path)
    print(f"✓ Datos cargados: {len(df):,} registros\n")
    
    # Preparar datos de uso
    print(" Analizando uso por estación...")
    uso_df = preparar_datos_uso(df)
    print(f"✓ Análisis completado para {len(uso_df)} estaciones\n")
    
    # Crear matriz temporal
    print("  Creando matriz temporal...")
    matriz_temporal = crear_matriz_temporal(df)
    print("✓ Matriz temporal creada\n")
    
    # Crear gráficos
    print(" Creando visualizaciones...")
    fig_main = crear_grafico(uso_df, matriz_temporal)
    fig_comp = crear_grafico_comparativo(uso_df)
    fig_efic = crear_grafico_eficiencia(uso_df)
    print("✓ Gráficos creados\n")
    
    # Analizar
    print("="*80)
    print(" ANÁLISIS DE USO POR ESTACIÓN")
    print("="*80)
    
    insights = analizar(uso_df, df)
    
    print(f"\n RESUMEN GENERAL:")
    print(f"   • Total de estaciones: {insights['total_estaciones']}")
    print(f"   • Total de transacciones: {insights['total_transacciones']:,}")
    print(f"   • Promedio por estación: {insights['promedio_trans']:.1f}")
    print(f"   • Mediana: {insights['mediana_trans']:.0f}")
    
    print(f"\n TOP 3 ESTACIONES:")
    print(f"   1. {insights['top_1_nombre']}: {insights['top_1_trans']:,} trans - ${insights['top_1_ingresos']:,} COP")
    print(f"   2. {insights['top_2_nombre']}: {insights['top_2_trans']:,} trans")
    print(f"   3. {insights['top_3_nombre']}: {insights['top_3_trans']:,} trans")
    print(f"   • Concentración: {insights['concentracion_top3']:.1f}% del total")
    
    print(f"\n⚡ EFICIENCIA:")
    print(f"   • Estación más eficiente: {insights['top_eficiencia_nombre']}")
    print(f"   • Transacciones por día: {insights['top_eficiencia_valor']:.2f}")
    
    print(f"\n BAJO RENDIMIENTO:")
    print(f"   • Estaciones con <50 trans: {insights['bajo_rendimiento_count']}")
    print(f"   • Porcentaje: {insights['bajo_rendimiento_pct']:.1f}%")
    
    print(f"\n ESTADÍSTICAS POR TIPO:")
    print(f"\n{'Tipo':<15} {'Estaciones':>12} {'Trans Total':>15} {'Trans Promedio':>17} {'Ingresos Total':>18}")
    print("-" * 85)
    for tipo in insights['stats_por_tipo'].index:
        row = insights['stats_por_tipo'].loc[tipo]
        print(f"{tipo:<15} {int(row[('evse_uid', 'count')]):>12} {int(row[('transacciones', 'sum')]):>15,} {row[('transacciones', 'mean')]:>17.1f} ${int(row[('ingresos_totales', 'sum')]):>17,}")
    
    print(f"\nTABLA COMPLETA (Top 10):")
    print(f"\n{'Estación':<30} {'Trans':>8} {'Usuarios':>10} {'Ingresos':>15} {'Trans/día':>10}")
    print("-" * 80)
    for _, row in uso_df.head(10).iterrows():
        print(f"{row['evse_uid']:<30} {int(row['transacciones']):>8} {int(row['usuarios_unicos']):>10} ${int(row['ingresos_totales']):>14,} {row['trans_por_dia']:>10.2f}")
    
    print(f"\n INSIGHTS PRINCIPALES:")
    print(f"\n   1. ESTACIÓN LÍDER:")
    print(f"      {insights['insight_top']}")
    
    print(f"\n   2. CONCENTRACIÓN:")
    print(f"      {insights['insight_concentracion']}")
    
    print(f"\n   3. BAJO RENDIMIENTO:")
    print(f"      {insights['insight_bajo_rendimiento']}")
    
    print(f"\n   4. EFICIENCIA:")
    print(f"      {insights['insight_eficiencia']}")
    
    print(f"\n   5. DISTRIBUCIÓN:")
    print(f"      {insights['insight_distribucion']}")
    
    print(f"\n   6. RECOMENDACIONES:")
    print(f"      • Optimizar las top 3 estaciones (alto ROI)")
    print(f"      • Evaluar estaciones de bajo rendimiento")
    print(f"      • Replicar prácticas de estaciones eficientes")
    print(f"      • Considerar expansión en ubicaciones tipo Éxito")
    
    # Guardar gráficos
    print("\n" + "="*80)
    print(" Guardando gráficos...")
    guardar_grafico(fig_main, 'grafico_07_heatmap_uso.png')
    guardar_grafico(fig_comp, 'grafico_07_comparativo.png')
    guardar_grafico(fig_efic, 'grafico_07_eficiencia.png')
    
    # Mostrar en navegador
    print(" Abriendo gráfico principal en navegador...")
    fig_main.show()
    
    print("\n Abriendo gráfico comparativo en navegador...")
    fig_comp.show()
    
    print("\n Abriendo gráfico de eficiencia en navegador...")
    fig_efic.show()
    
    