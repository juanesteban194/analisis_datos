"""
Gráfico 5: Ingresos Mensuales
Análisis: Evolución temporal de ingresos con tendencias y estacionalidad
Tipo: Gráfico de líneas con área y tendencia
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
import sys
from datetime import datetime

# Agregar el directorio raíz al path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from utils import cargar_datos, guardar_grafico, formatear_moneda

def preparar_datos_temporales(df):
    """
    Prepara los datos agregados por mes
    
    Args:
        df: DataFrame con datos de Oasis
    
    Returns:
        DataFrame con métricas mensuales
    """
    
    # Crear columna de año-mes
    df['year_month'] = df['start_date_time'].dt.to_period('M')
    df['year_month_str'] = df['year_month'].astype(str)
    
    # Agregar por mes
    monthly_data = df.groupby('year_month_str').agg({
        'amount_transaction': ['sum', 'mean', 'count'],
        'user_id': 'nunique',
        'evse_uid': 'nunique',
        'energy_kwh': 'sum'
    }).reset_index()
    
    # Aplanar columnas multi-nivel
    monthly_data.columns = ['year_month', 'ingresos_totales', 'ingreso_promedio', 
                            'num_transacciones', 'usuarios_activos', 
                            'estaciones_activas', 'energia_total_kwh']
    
    # Calcular métricas adicionales
    monthly_data['ingreso_por_usuario'] = (
        monthly_data['ingresos_totales'] / monthly_data['usuarios_activos']
    )
    
    monthly_data['ingreso_por_transaccion'] = (
        monthly_data['ingresos_totales'] / monthly_data['num_transacciones']
    )
    
    # Calcular crecimiento mes a mes
    monthly_data['crecimiento_ingresos_pct'] = (
        monthly_data['ingresos_totales'].pct_change() * 100
    )
    
    monthly_data['crecimiento_usuarios_pct'] = (
        monthly_data['usuarios_activos'].pct_change() * 100
    )
    
    # Ordenar por fecha
    monthly_data = monthly_data.sort_values('year_month')
    
    return monthly_data

def calcular_tendencia(data, columna):
    """
    Calcula la línea de tendencia usando regresión lineal
    
    Args:
        data: DataFrame con datos
        columna: Nombre de la columna para calcular tendencia
    
    Returns:
        Array con valores de tendencia
    """
    x = np.arange(len(data))
    y = data[columna].values
    
    # Regresión lineal simple
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    
    return p(x)

def crear_grafico(monthly_data):
    """
    Crea gráfico de líneas con evolución temporal de ingresos
    
    Args:
        monthly_data: DataFrame con datos mensuales
    
    Returns:
        Figure de Plotly
    """
    
    # Calcular tendencia
    tendencia_ingresos = calcular_tendencia(monthly_data, 'ingresos_totales')
    
    # Crear subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            'Evolución de Ingresos Mensuales',
            'Usuarios Activos y Transacciones por Mes',
            'Crecimiento Mensual (%)'
        ),
        vertical_spacing=0.1,
        row_heights=[0.4, 0.3, 0.3]
    )
    
    # Gráfico 1: Ingresos con área y tendencia
    fig.add_trace(
        go.Scatter(
            x=monthly_data['year_month'],
            y=monthly_data['ingresos_totales'],
            mode='lines+markers',
            name='Ingresos',
            line=dict(color='#3b82f6', width=3),
            marker=dict(size=8, color='#3b82f6'),
            fill='tozeroy',
            fillcolor='rgba(59, 130, 246, 0.2)',
            hovertemplate='<b>%{x}</b><br>Ingresos: $%{y:,.0f} COP<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Línea de tendencia
    fig.add_trace(
        go.Scatter(
            x=monthly_data['year_month'],
            y=tendencia_ingresos,
            mode='lines',
            name='Tendencia',
            line=dict(color='#ef4444', width=2, dash='dash'),
            hovertemplate='<b>%{x}</b><br>Tendencia: $%{y:,.0f} COP<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Gráfico 2: Usuarios y Transacciones
    fig.add_trace(
        go.Scatter(
            x=monthly_data['year_month'],
            y=monthly_data['usuarios_activos'],
            mode='lines+markers',
            name='Usuarios Activos',
            line=dict(color='#10b981', width=2),
            marker=dict(size=6),
            yaxis='y3',
            hovertemplate='<b>%{x}</b><br>Usuarios: %{y}<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=monthly_data['year_month'],
            y=monthly_data['num_transacciones'],
            mode='lines+markers',
            name='Transacciones',
            line=dict(color='#8b5cf6', width=2),
            marker=dict(size=6),
            yaxis='y4',
            hovertemplate='<b>%{x}</b><br>Transacciones: %{y}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Gráfico 3: Crecimiento
    colors = ['#10b981' if x >= 0 else '#ef4444' 
              for x in monthly_data['crecimiento_ingresos_pct'].fillna(0)]
    
    fig.add_trace(
        go.Bar(
            x=monthly_data['year_month'],
            y=monthly_data['crecimiento_ingresos_pct'].fillna(0),
            name='Crecimiento Ingresos',
            marker_color=colors,
            hovertemplate='<b>%{x}</b><br>Crecimiento: %{y:.1f}%<extra></extra>'
        ),
        row=3, col=1
    )
    
    # Layout
    fig.update_layout(
        title={
            'text': 'Análisis Temporal de Ingresos - Evolución Mensual',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2d3748', 'family': 'Arial Black'}
        },
        showlegend=True,
        height=1000,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Arial', size=12),
        hovermode='x unified'
    )
    
    # Actualizar ejes
    fig.update_xaxes(
        tickangle=-45,
        gridcolor='#e2e8f0',
        title_text="Mes",
        row=3, col=1
    )
    
    fig.update_yaxes(
        title_text="Ingresos (COP)",
        gridcolor='#e2e8f0',
        row=1, col=1
    )
    
    fig.update_yaxes(
        title_text="Usuarios / Transacciones",
        gridcolor='#e2e8f0',
        row=2, col=1
    )
    
    fig.update_yaxes(
        title_text="Crecimiento (%)",
        gridcolor='#e2e8f0',
        zeroline=True,
        zerolinecolor='#94a3b8',
        row=3, col=1
    )
    
    return fig

def crear_grafico_comparativo(monthly_data):
    """
    Crea gráfico comparativo de métricas normalizadas
    
    Args:
        monthly_data: DataFrame con datos mensuales
    
    Returns:
        Figure de Plotly
    """
    
    # Normalizar métricas (0-100)
    def normalize(series):
        return (series - series.min()) / (series.max() - series.min()) * 100
    
    fig = go.Figure()
    
    # Ingresos normalizados
    fig.add_trace(go.Scatter(
        x=monthly_data['year_month'],
        y=normalize(monthly_data['ingresos_totales']),
        mode='lines+markers',
        name='Ingresos',
        line=dict(color='#3b82f6', width=2)
    ))
    
    # Usuarios normalizados
    fig.add_trace(go.Scatter(
        x=monthly_data['year_month'],
        y=normalize(monthly_data['usuarios_activos']),
        mode='lines+markers',
        name='Usuarios',
        line=dict(color='#10b981', width=2)
    ))
    
    # Transacciones normalizadas
    fig.add_trace(go.Scatter(
        x=monthly_data['year_month'],
        y=normalize(monthly_data['num_transacciones']),
        mode='lines+markers',
        name='Transacciones',
        line=dict(color='#8b5cf6', width=2)
    ))
    
    fig.update_layout(
        title={
            'text': 'Comparación de Métricas Normalizadas (0-100)',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#2d3748', 'family': 'Arial Black'}
        },
        xaxis_title="Mes",
        yaxis_title="Índice Normalizado (0-100)",
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=500,
        font=dict(family='Arial', size=12),
        hovermode='x unified'
    )
    
    fig.update_xaxes(tickangle=-45, gridcolor='#e2e8f0')
    fig.update_yaxes(gridcolor='#e2e8f0')
    
    return fig

def analizar(monthly_data):
    """
    Analiza tendencias temporales y genera insights
    
    Args:
        monthly_data: DataFrame con datos mensuales
    
    Returns:
        dict con insights
    """
    
    # Totales
    total_ingresos = monthly_data['ingresos_totales'].sum()
    total_transacciones = monthly_data['num_transacciones'].sum()
    
    # Promedios
    ingreso_promedio_mes = monthly_data['ingresos_totales'].mean()
    usuarios_promedio_mes = monthly_data['usuarios_activos'].mean()
    
    # Mejor y peor mes
    mejor_mes = monthly_data.loc[monthly_data['ingresos_totales'].idxmax()]
    peor_mes = monthly_data.loc[monthly_data['ingresos_totales'].idxmin()]
    
    # Crecimiento total
    primer_mes = monthly_data.iloc[0]['ingresos_totales']
    ultimo_mes = monthly_data.iloc[-1]['ingresos_totales']
    crecimiento_total = ((ultimo_mes - primer_mes) / primer_mes * 100)
    
    # Tendencia (pendiente)
    x = np.arange(len(monthly_data))
    y = monthly_data['ingresos_totales'].values
    z = np.polyfit(x, y, 1)
    tendencia_pendiente = z[0]
    
    # Volatilidad
    volatilidad = monthly_data['ingresos_totales'].std() / ingreso_promedio_mes * 100
    
    # Meses con crecimiento positivo
    meses_positivos = (monthly_data['crecimiento_ingresos_pct'] > 0).sum()
    total_meses = len(monthly_data) - 1  # -1 porque el primer mes no tiene crecimiento
    
    insights = {
        'total_meses': len(monthly_data),
        'total_ingresos': int(total_ingresos),
        'total_transacciones': int(total_transacciones),
        
        'ingreso_promedio_mes': int(ingreso_promedio_mes),
        'usuarios_promedio_mes': int(usuarios_promedio_mes),
        
        'mejor_mes_fecha': mejor_mes['year_month'],
        'mejor_mes_ingresos': int(mejor_mes['ingresos_totales']),
        
        'peor_mes_fecha': peor_mes['year_month'],
        'peor_mes_ingresos': int(peor_mes['ingresos_totales']),
        
        'crecimiento_total': crecimiento_total,
        'tendencia': 'positiva' if tendencia_pendiente > 0 else 'negativa',
        'tendencia_valor': tendencia_pendiente,
        
        'volatilidad': volatilidad,
        
        'meses_crecimiento_positivo': meses_positivos,
        'pct_meses_positivos': (meses_positivos / total_meses * 100) if total_meses > 0 else 0,
        
        'insight_tendencia': f"La tendencia general es {('positiva' if tendencia_pendiente > 0 else 'negativa')} con un crecimiento {'sostenido' if crecimiento_total > 0 else 'negativo'} de {crecimiento_total:.1f}% desde el inicio.",
        
        'insight_mejor_peor': f"El mejor mes fue {mejor_mes['year_month']} (${mejor_mes['ingresos_totales']:,.0f} COP) y el peor fue {peor_mes['year_month']} (${peor_mes['ingresos_totales']:,.0f} COP), una diferencia de {((mejor_mes['ingresos_totales'] - peor_mes['ingresos_totales']) / peor_mes['ingresos_totales'] * 100):.1f}%.",
        
        'insight_volatilidad': f"La volatilidad de ingresos es de {volatilidad:.1f}%, indicando {'alta' if volatilidad > 30 else 'moderada' if volatilidad > 15 else 'baja'} variabilidad mensual.",
        
        'insight_consistencia': f"{meses_positivos} de {total_meses} meses ({(meses_positivos/total_meses*100):.1f}%) mostraron crecimiento positivo, {'demostrando consistencia' if (meses_positivos/total_meses) > 0.5 else 'indicando inconsistencia'}."
    }
    
    return insights

if __name__ == "__main__":
    print("\n" + "="*80)
    print("GRÁFICO 5: INGRESOS MENSUALES")
    print("="*80)
    
    # Cargar datos
    csv_path = os.path.join(parent_dir, 'data', 'df_oasis_clean.csv')
    df = cargar_datos(csv_path)
    print(f"✓ Datos cargados: {len(df):,} registros\n")
    
    # Preparar datos temporales
    print("📅 Preparando datos mensuales...")
    monthly_data = preparar_datos_temporales(df)
    print(f"✓ Datos agregados para {len(monthly_data)} meses\n")
    
    # Crear gráficos
    print("📊 Creando gráfico principal...")
    fig_main = crear_grafico(monthly_data)
    
    print("📊 Creando gráfico comparativo...")
    fig_comp = crear_grafico_comparativo(monthly_data)
    
    # Analizar
    print("\n" + "="*80)
    print("📊 ANÁLISIS TEMPORAL")
    print("="*80)
    
    insights = analizar(monthly_data)
    
    print(f"\n📈 RESUMEN GENERAL:")
    print(f"   • Período analizado: {insights['total_meses']} meses")
    print(f"   • Ingresos totales: ${insights['total_ingresos']:,} COP")
    print(f"   • Transacciones totales: {insights['total_transacciones']:,}")
    print(f"   • Ingreso promedio/mes: ${insights['ingreso_promedio_mes']:,} COP")
    print(f"   • Usuarios promedio/mes: {insights['usuarios_promedio_mes']}")
    
    print(f"\n🏆 MEJOR Y PEOR MES:")
    print(f"   • Mejor: {insights['mejor_mes_fecha']} - ${insights['mejor_mes_ingresos']:,} COP")
    print(f"   • Peor: {insights['peor_mes_fecha']} - ${insights['peor_mes_ingresos']:,} COP")
    
    print(f"\n📊 TENDENCIA:")
    print(f"   • Crecimiento total: {insights['crecimiento_total']:.1f}%")
    print(f"   • Tendencia: {insights['tendencia'].upper()}")
    print(f"   • Volatilidad: {insights['volatilidad']:.1f}%")
    
    print(f"\n✅ CONSISTENCIA:")
    print(f"   • Meses con crecimiento: {insights['meses_crecimiento_positivo']} de {insights['total_meses']-1}")
    print(f"   • Porcentaje positivo: {insights['pct_meses_positivos']:.1f}%")
    
    print(f"\n📋 TABLA MENSUAL:")
    print(f"\n{'Mes':<12} {'Ingresos':>15} {'Usuarios':>10} {'Trans':>8} {'Crec %':>8}")
    print("-" * 60)
    for _, row in monthly_data.iterrows():
        crec = row['crecimiento_ingresos_pct']
        crec_str = f"{crec:>7.1f}%" if not pd.isna(crec) else "    -   "
        print(f"{row['year_month']:<12} ${row['ingresos_totales']:>14,.0f} {int(row['usuarios_activos']):>10} {int(row['num_transacciones']):>8} {crec_str}")
    
    print(f"\n💡 INSIGHTS PRINCIPALES:")
    print(f"\n   1. TENDENCIA GENERAL:")
    print(f"      {insights['insight_tendencia']}")
    
    print(f"\n   2. MEJOR Y PEOR DESEMPEÑO:")
    print(f"      {insights['insight_mejor_peor']}")
    
    print(f"\n   3. VOLATILIDAD:")
    print(f"      {insights['insight_volatilidad']}")
    
    print(f"\n   4. CONSISTENCIA:")
    print(f"      {insights['insight_consistencia']}")
    
    print(f"\n   5. RECOMENDACIONES:")
    if insights['tendencia'] == 'positiva':
        print(f"      • Mantener estrategias actuales que impulsan el crecimiento")
        print(f"      • Capitalizar en los meses de mejor desempeño")
    else:
        print(f"      • Revisar estrategias: la tendencia es negativa")
        print(f"      • Implementar campañas de recuperación")
    
    if insights['volatilidad'] > 30:
        print(f"      • Alta volatilidad: estabilizar flujo de ingresos")
        print(f"      • Identificar factores estacionales")
    
    # Guardar gráficos
    print("\n" + "="*80)
    print("💾 Guardando gráficos...")
    guardar_grafico(fig_main, 'grafico_05_ingresos_mensuales.png')
    guardar_grafico(fig_comp, 'grafico_05_comparativo.png')
    
    # Mostrar en navegador
    print("🌐 Abriendo gráfico principal en navegador...")
    fig_main.show()
    
    print("\n🌐 Abriendo gráfico comparativo en navegador...")
    fig_comp.show()
    
   