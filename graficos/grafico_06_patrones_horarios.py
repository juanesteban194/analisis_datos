"""
Gráfico 6: Patrones Horarios
Análisis: Heatmap de uso de estaciones por hora del día y día de la semana
Tipo: Heatmap interactivo con análisis de patrones temporales
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

def preparar_datos_temporales(df):
    """
    Prepara datos con información temporal (hora, día de semana)
    
    Args:
        df: DataFrame con datos de Oasis
    
    Returns:
        DataFrame con columnas temporales
    """
    
    # Extraer información temporal
    df['hora'] = df['start_date_time'].dt.hour
    df['dia_semana'] = df['start_date_time'].dt.dayofweek
    df['dia_nombre'] = df['start_date_time'].dt.day_name()
    df['es_fin_semana'] = df['dia_semana'].isin([5, 6])
    
    # Mapeo de días en español
    dias_map = {
        'Monday': 'Lunes',
        'Tuesday': 'Martes',
        'Wednesday': 'Miércoles',
        'Thursday': 'Jueves',
        'Friday': 'Viernes',
        'Saturday': 'Sábado',
        'Sunday': 'Domingo'
    }
    df['dia_nombre_es'] = df['dia_nombre'].map(dias_map)
    
    return df

def crear_matriz_heatmap(df):
    """
    Crea matriz de transacciones por hora y día
    
    Args:
        df: DataFrame con datos temporales
    
    Returns:
        DataFrame pivotado para heatmap
    """
    
    # Orden correcto de días
    dias_orden = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    
    # Agregar por hora y día
    heatmap_data = df.groupby(['hora', 'dia_nombre_es']).size().reset_index(name='transacciones')
    
    # Pivotar para matriz
    matriz = heatmap_data.pivot(index='hora', columns='dia_nombre_es', values='transacciones')
    
    # Reordenar columnas
    matriz = matriz.reindex(columns=dias_orden, fill_value=0)
    
    return matriz

def crear_grafico(df, matriz):
    """
    Crea heatmap de uso por hora y día
    
    Args:
        df: DataFrame con datos
        matriz: Matriz pivotada para heatmap
    
    Returns:
        Figure de Plotly
    """
    
    # Crear figura con subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Heatmap de Uso: Hora vs Día de la Semana',
            'Transacciones por Hora del Día',
            'Transacciones por Día de la Semana',
            'Comparación: Días Laborales vs Fin de Semana'
        ),
        specs=[
            [{"type": "heatmap", "colspan": 2}, None],
            [{"type": "bar"}, {"type": "bar"}]
        ],
        row_heights=[0.6, 0.4],
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )
    
    # Heatmap principal
    fig.add_trace(
        go.Heatmap(
            z=matriz.values,
            x=matriz.columns,
            y=matriz.index,
            colorscale='Blues',
            hovertemplate='<b>%{x}</b><br>Hora: %{y}:00<br>Transacciones: %{z}<extra></extra>',
            colorbar=dict(title="Transacciones", x=1.02)
        ),
        row=1, col=1
    )
    
    # Transacciones por hora
    trans_por_hora = df.groupby('hora').size()
    fig.add_trace(
        go.Bar(
            x=trans_por_hora.index,
            y=trans_por_hora.values,
            marker_color='#3b82f6',
            name='Por Hora',
            hovertemplate='<b>Hora: %{x}:00</b><br>Transacciones: %{y}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Transacciones por día
    dias_orden = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    trans_por_dia = df.groupby('dia_nombre_es').size().reindex(dias_orden, fill_value=0)
    
    colors_dias = ['#3b82f6'] * 5 + ['#10b981'] * 2  # Azul laborales, verde fin de semana
    
    fig.add_trace(
        go.Bar(
            x=trans_por_dia.index,
            y=trans_por_dia.values,
            marker_color=colors_dias,
            name='Por Día',
            hovertemplate='<b>%{x}</b><br>Transacciones: %{y}<extra></extra>'
        ),
        row=2, col=2
    )
    
    # Layout
    fig.update_layout(
        title={
            'text': 'Análisis de Patrones Horarios de Uso',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2d3748', 'family': 'Arial Black'}
        },
        showlegend=False,
        height=1000,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Arial', size=11)
    )
    
    # Actualizar ejes
    fig.update_xaxes(title_text="Día de la Semana", row=1, col=1)
    fig.update_yaxes(title_text="Hora del Día", row=1, col=1)
    
    fig.update_xaxes(title_text="Hora", row=2, col=1)
    fig.update_yaxes(title_text="Transacciones", gridcolor='#e2e8f0', row=2, col=1)
    
    fig.update_xaxes(title_text="Día", tickangle=-45, row=2, col=2)
    fig.update_yaxes(title_text="Transacciones", gridcolor='#e2e8f0', row=2, col=2)
    
    return fig

def crear_grafico_horario_detallado(df):
    """
    Crea gráfico de líneas por hora con días separados
    
    Args:
        df: DataFrame con datos
    
    Returns:
        Figure de Plotly
    """
    
    dias_orden = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    
    fig = go.Figure()
    
    # Línea para cada día
    for dia in dias_orden:
        df_dia = df[df['dia_nombre_es'] == dia]
        trans_por_hora = df_dia.groupby('hora').size()
        
        # Color diferente para fin de semana
        color = '#10b981' if dia in ['Sábado', 'Domingo'] else '#3b82f6'
        width = 3 if dia in ['Sábado', 'Domingo'] else 2
        
        fig.add_trace(go.Scatter(
            x=trans_por_hora.index,
            y=trans_por_hora.values,
            mode='lines+markers',
            name=dia,
            line=dict(color=color, width=width),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title={
            'text': 'Patrones Horarios por Día de la Semana',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#2d3748', 'family': 'Arial Black'}
        },
        xaxis_title="Hora del Día",
        yaxis_title="Número de Transacciones",
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=600,
        font=dict(family='Arial', size=12),
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_xaxes(gridcolor='#e2e8f0', range=[0, 23])
    fig.update_yaxes(gridcolor='#e2e8f0')
    
    return fig

def analizar(df, matriz):
    """
    Analiza patrones horarios y genera insights
    
    Args:
        df: DataFrame con datos temporales
        matriz: Matriz de heatmap
    
    Returns:
        dict con insights
    """
    
    # Hora pico
    trans_por_hora = df.groupby('hora').size()
    hora_pico = trans_por_hora.idxmax()
    trans_hora_pico = trans_por_hora.max()
    
    # Hora valle
    hora_valle = trans_por_hora.idxmin()
    trans_hora_valle = trans_por_hora.min()
    
    # Día más activo
    dias_orden = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    trans_por_dia = df.groupby('dia_nombre_es').size().reindex(dias_orden)
    dia_pico = trans_por_dia.idxmax()
    trans_dia_pico = trans_por_dia.max()
    
    # Comparación laborales vs fin de semana
    trans_laborales = df[~df['es_fin_semana']].shape[0]
    trans_fin_semana = df[df['es_fin_semana']].shape[0]
    pct_laborales = trans_laborales / len(df) * 100
    pct_fin_semana = trans_fin_semana / len(df) * 100
    
    # Promedio por día de cada tipo
    dias_laborales = 5
    dias_fin_semana = 2
    promedio_dia_laboral = trans_laborales / dias_laborales
    promedio_dia_fin_semana = trans_fin_semana / dias_fin_semana
    
    # Horarios específicos
    trans_manana = df[df['hora'].between(6, 11)].shape[0]  # 6am-11am
    trans_mediodia = df[df['hora'].between(12, 14)].shape[0]  # 12pm-2pm
    trans_tarde = df[df['hora'].between(15, 19)].shape[0]  # 3pm-7pm
    trans_noche = df[df['hora'].between(20, 23)].shape[0]  # 8pm-11pm
    trans_madrugada = df[df['hora'].between(0, 5)].shape[0]  # 12am-5am
    
    total = len(df)
    
    # Identificar punto más activo (día + hora)
    punto_max = matriz.stack().idxmax()
    punto_max_valor = matriz.stack().max()
    
    insights = {
        'total_transacciones': total,
        
        'hora_pico': hora_pico,
        'trans_hora_pico': int(trans_hora_pico),
        'pct_hora_pico': (trans_hora_pico / total * 100),
        
        'hora_valle': hora_valle,
        'trans_hora_valle': int(trans_hora_valle),
        
        'dia_pico': dia_pico,
        'trans_dia_pico': int(trans_dia_pico),
        
        'trans_laborales': trans_laborales,
        'pct_laborales': pct_laborales,
        'promedio_dia_laboral': promedio_dia_laboral,
        
        'trans_fin_semana': trans_fin_semana,
        'pct_fin_semana': pct_fin_semana,
        'promedio_dia_fin_semana': promedio_dia_fin_semana,
        
        'trans_manana': trans_manana,
        'pct_manana': (trans_manana / total * 100),
        
        'trans_mediodia': trans_mediodia,
        'pct_mediodia': (trans_mediodia / total * 100),
        
        'trans_tarde': trans_tarde,
        'pct_tarde': (trans_tarde / total * 100),
        
        'trans_noche': trans_noche,
        'pct_noche': (trans_noche / total * 100),
        
        'trans_madrugada': trans_madrugada,
        'pct_madrugada': (trans_madrugada / total * 100),
        
        'punto_max_hora': punto_max[0],
        'punto_max_dia': punto_max[1],
        'punto_max_valor': int(punto_max_valor),
        
        'insight_hora_pico': f"La hora pico es {hora_pico}:00 con {trans_hora_pico} transacciones ({(trans_hora_pico/total*100):.1f}% del total).",
        
        'insight_dia_pico': f"El día más activo es {dia_pico} con {trans_dia_pico} transacciones.",
        
        'insight_laboral_vs_finde': f"Los días laborales concentran {pct_laborales:.1f}% de las transacciones vs {pct_fin_semana:.1f}% en fin de semana. Promedio: {promedio_dia_laboral:.0f} trans/día laboral vs {promedio_dia_fin_semana:.0f} trans/día fin de semana.",
        
        'insight_franjas': f"La franja más activa es la {'mañana' if trans_manana == max(trans_manana, trans_mediodia, trans_tarde, trans_noche) else 'tarde' if trans_tarde == max(trans_manana, trans_mediodia, trans_tarde, trans_noche) else 'mediodía'} ({max(trans_manana, trans_mediodia, trans_tarde, trans_noche)} transacciones).",
        
        'insight_punto_caliente': f"El punto más activo es {punto_max[1]} a las {punto_max[0]}:00 con {punto_max_valor} transacciones."
    }
    
    return insights

if __name__ == "__main__":
    print("\n" + "="*80)
    print("GRÁFICO 6: PATRONES HORARIOS")
    print("="*80)
    
    # Cargar datos
    csv_path = os.path.join(parent_dir, 'data', 'df_oasis_clean.csv')
    df = cargar_datos(csv_path)
    print(f"✓ Datos cargados: {len(df):,} registros\n")
    
    # Preparar datos temporales
    print("⏰ Preparando datos temporales...")
    df = preparar_datos_temporales(df)
    print("✓ Datos temporales listos\n")
    
    # Crear matriz heatmap
    print("🔥 Creando matriz de heatmap...")
    matriz = crear_matriz_heatmap(df)
    print("✓ Matriz creada\n")
    
    # Crear gráficos
    print("📊 Creando heatmap principal...")
    fig_main = crear_grafico(df, matriz)
    
    print("📊 Creando gráfico horario detallado...")
    fig_detail = crear_grafico_horario_detallado(df)
    
    # Analizar
    print("\n" + "="*80)
    print("📊 ANÁLISIS DE PATRONES HORARIOS")
    print("="*80)
    
    insights = analizar(df, matriz)
    
    print(f"\n⏰ PATRONES POR HORA:")
    print(f"   • Hora pico: {insights['hora_pico']}:00")
    print(f"   • Transacciones en hora pico: {insights['trans_hora_pico']} ({insights['pct_hora_pico']:.1f}%)")
    print(f"   • Hora valle: {insights['hora_valle']}:00")
    print(f"   • Transacciones en hora valle: {insights['trans_hora_valle']}")
    
    print(f"\n📅 PATRONES POR DÍA:")
    print(f"   • Día más activo: {insights['dia_pico']}")
    print(f"   • Transacciones: {insights['trans_dia_pico']}")
    
    print(f"\n💼 LABORALES VS FIN DE SEMANA:")
    print(f"   • Días laborales: {insights['trans_laborales']:,} trans ({insights['pct_laborales']:.1f}%)")
    print(f"   • Promedio/día laboral: {insights['promedio_dia_laboral']:.0f}")
    print(f"   • Fin de semana: {insights['trans_fin_semana']:,} trans ({insights['pct_fin_semana']:.1f}%)")
    print(f"   • Promedio/día fin de semana: {insights['promedio_dia_fin_semana']:.0f}")
    
    print(f"\n🕐 FRANJAS HORARIAS:")
    print(f"   • Mañana (6am-11am):   {insights['trans_manana']:>5,} trans ({insights['pct_manana']:>5.1f}%)")
    print(f"   • Mediodía (12pm-2pm):  {insights['trans_mediodia']:>5,} trans ({insights['pct_mediodia']:>5.1f}%)")
    print(f"   • Tarde (3pm-7pm):      {insights['trans_tarde']:>5,} trans ({insights['pct_tarde']:>5.1f}%)")
    print(f"   • Noche (8pm-11pm):     {insights['trans_noche']:>5,} trans ({insights['pct_noche']:>5.1f}%)")
    print(f"   • Madrugada (12am-5am): {insights['trans_madrugada']:>5,} trans ({insights['pct_madrugada']:>5.1f}%)")
    
    print(f"\n🔥 PUNTO MÁS CALIENTE:")
    print(f"   • Día: {insights['punto_max_dia']}")
    print(f"   • Hora: {insights['punto_max_hora']}:00")
    print(f"   • Transacciones: {insights['punto_max_valor']}")
    
    print(f"\n💡 INSIGHTS PRINCIPALES:")
    print(f"\n   1. HORA PICO:")
    print(f"      {insights['insight_hora_pico']}")
    
    print(f"\n   2. DÍA MÁS ACTIVO:")
    print(f"      {insights['insight_dia_pico']}")
    
    print(f"\n   3. PATRÓN SEMANAL:")
    print(f"      {insights['insight_laboral_vs_finde']}")
    
    print(f"\n   4. FRANJAS HORARIAS:")
    print(f"      {insights['insight_franjas']}")
    
    print(f"\n   5. PUNTO CALIENTE:")
    print(f"      {insights['insight_punto_caliente']}")
    
    print(f"\n   6. RECOMENDACIONES OPERATIVAS:")
    print(f"      • Aumentar capacidad en hora pico ({insights['hora_pico']}:00)")
    print(f"      • Mantenimiento programado en horas valle ({insights['hora_valle']}:00)")
    print(f"      • Promociones en horarios de baja demanda")
    print(f"      • Personal adicional los {insights['dia_pico']}")
    
    # Guardar gráficos
    print("\n" + "="*80)
    print("💾 Guardando gráficos...")
    guardar_grafico(fig_main, 'grafico_06_patrones_horarios.png')
    guardar_grafico(fig_detail, 'grafico_06_horario_detallado.png')
    
    # Mostrar en navegador
    print("🌐 Abriendo heatmap en navegador...")
    fig_main.show()
    
    print("\n🌐 Abriendo gráfico detallado en navegador...")
    fig_detail.show()
    
    