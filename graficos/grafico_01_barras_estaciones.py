"""
Gráfico 1: Barras de Estaciones - ACTUALIZADO CON DATOS NORMALIZADOS
Análisis: Transacciones por estación con datos monetarios correctos
Tipo: Gráfico de barras vertical
"""

import pandas as pd
import plotly.graph_objects as go
import os
import sys

# Agregar el directorio raíz al path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from utils import cargar_datos, guardar_grafico

def crear_grafico(df):
    """
    Crea el gráfico de barras con transacciones por estación
    
    Args:
        df: DataFrame con datos de Oasis
    
    Returns:
        Figure de Plotly
    """
    
    # Contar transacciones por estación
    df_count = df.groupby('evse_uid').size().reset_index(name='transacciones')
    df_count = df_count.sort_values('transacciones', ascending=False)
    
    # Crear gráfico de barras
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_count['evse_uid'],
        y=df_count['transacciones'],
        marker=dict(
            color=df_count['transacciones'],
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title="Transacciones")
        ),
        text=df_count['transacciones'],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Transacciones: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': 'Uso de Estaciones de Carga - Transacciones por Estación',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2d3748', 'family': 'Arial Black'}
        },
        xaxis_title="Estación (evse_uid)",
        yaxis_title="Número de Transacciones",
        xaxis={
            'tickangle': -45,
            'tickfont': {'size': 10}
        },
        yaxis={
            'gridcolor': '#e2e8f0',
            'zeroline': True
        },
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=600,
        margin=dict(b=150),
        hovermode='x unified',
        font=dict(family='Arial', size=12)
    )
    
    return fig

def analizar(df):
    """
    Analiza los datos y retorna insights clave
    
    Args:
        df: DataFrame con datos
    
    Returns:
        dict con métricas e insights
    """
    
    # Contar transacciones por estación
    df_count = df.groupby('evse_uid').size().sort_values(ascending=False)
    
    # Calcular estadísticas
    total_estaciones = len(df_count)
    total_transacciones = df_count.sum()
    promedio = df_count.mean()
    mediana = df_count.median()
    
    # Calcular ingresos por estación
    ingresos_estacion = df.groupby('evse_uid')['amount_transaction'].sum().sort_values(ascending=False)
    ingresos_totales = ingresos_estacion.sum()
    
    # Top 3 estaciones
    top_3 = df_count.head(3)
    top_3_pct = (top_3.sum() / total_transacciones * 100)
    
    # Ingresos top 3
    top_3_ingresos = ingresos_estacion.head(3)
    top_3_ingresos_pct = (top_3_ingresos.sum() / ingresos_totales * 100)
    
    # Estaciones de bajo rendimiento (<50 transacciones)
    bajo_rendimiento = df_count[df_count < 50]
    
    # Identificar estaciones Éxito vs otras
    estaciones_exito = df_count[df_count.index.str.contains('Exito', case=False, na=False)]
    pct_exito = (estaciones_exito.sum() / total_transacciones * 100)
    
    # Ingresos Éxito
    ingresos_exito = ingresos_estacion[ingresos_estacion.index.str.contains('Exito', case=False, na=False)]
    pct_ingresos_exito = (ingresos_exito.sum() / ingresos_totales * 100)
    
    insights = {
        'total_estaciones': total_estaciones,
        'total_transacciones': int(total_transacciones),
        'ingresos_totales': int(ingresos_totales),
        'promedio_transacciones': f"{promedio:.1f}",
        'mediana_transacciones': int(mediana),
        
        'top_1_nombre': top_3.index[0],
        'top_1_valor': int(top_3.iloc[0]),
        'top_1_ingresos': int(ingresos_estacion[top_3.index[0]]),
        
        'top_2_nombre': top_3.index[1],
        'top_2_valor': int(top_3.iloc[1]),
        'top_2_ingresos': int(ingresos_estacion[top_3.index[1]]),
        
        'top_3_nombre': top_3.index[2],
        'top_3_valor': int(top_3.iloc[2]),
        'top_3_ingresos': int(ingresos_estacion[top_3.index[2]]),
        
        'top_3_concentracion': f"{top_3_pct:.1f}%",
        'top_3_ingresos_concentracion': f"{top_3_ingresos_pct:.1f}%",
        
        'estaciones_bajo_rendimiento': len(bajo_rendimiento),
        'pct_bajo_rendimiento': f"{(len(bajo_rendimiento)/total_estaciones*100):.1f}%",
        
        'estaciones_exito_count': len(estaciones_exito),
        'pct_transacciones_exito': f"{pct_exito:.1f}%",
        'pct_ingresos_exito': f"{pct_ingresos_exito:.1f}%",
        'ingresos_exito': int(ingresos_exito.sum()),
        
        'insight_principal': f"Las 3 estaciones top ({top_3.index[0]}, {top_3.index[1]}, {top_3.index[2]}) concentran el {top_3_pct:.1f}% de transacciones y {top_3_ingresos_pct:.1f}% de ingresos.",
        
        'insight_exito': f"Las estaciones 'Éxito' ({len(estaciones_exito)} estaciones) generan el {pct_exito:.1f}% de transacciones y ${int(ingresos_exito.sum()):,} COP ({pct_ingresos_exito:.1f}% del total).",
        
        'insight_bajo_rendimiento': f"{len(bajo_rendimiento)} estaciones ({(len(bajo_rendimiento)/total_estaciones*100):.1f}%) tienen menos de 50 transacciones. Son candidatas para evaluación o reubicación."
    }
    
    return insights

if __name__ == "__main__":
    print("\n" + "="*80)
    print("GRÁFICO 1: TRANSACCIONES POR ESTACIÓN (DATOS NORMALIZADOS)")
    print("="*80)
    
    # Cargar datos
    csv_path = 'data/df_oasis_clean.csv'
    df = cargar_datos(csv_path)
    print(f"✓ Datos cargados: {len(df):,} registros")
    print(f"✓ Datos normalizados - Transacción promedio: ${df['amount_transaction'].mean():,.0f} COP\n")
    
    # Crear gráfico
    print(" Creando gráfico...")
    fig = crear_grafico(df)
    
    # Analizar datos
    print("\n" + "="*80)
    print(" ANÁLISIS E INSIGHTS")
    print("="*80)
    
    insights = analizar(df)
    
    print(f"\n MÉTRICAS GENERALES:")
    print(f"   • Total de estaciones: {insights['total_estaciones']}")
    print(f"   • Total de transacciones: {insights['total_transacciones']:,}")
    print(f"   • Ingresos totales: ${insights['ingresos_totales']:,} COP")
    print(f"   • Promedio por estación: {insights['promedio_transacciones']}")
    print(f"   • Mediana: {insights['mediana_transacciones']}")
    
    print(f"\n TOP 3 ESTACIONES:")
    print(f"   1. {insights['top_1_nombre']}: {insights['top_1_valor']:,} trans → ${insights['top_1_ingresos']:,} COP")
    print(f"   2. {insights['top_2_nombre']}: {insights['top_2_valor']:,} trans → ${insights['top_2_ingresos']:,} COP")
    print(f"   3. {insights['top_3_nombre']}: {insights['top_3_valor']:,} trans → ${insights['top_3_ingresos']:,} COP")
    print(f"   • Concentración: {insights['top_3_concentracion']} transacciones, {insights['top_3_ingresos_concentracion']} ingresos")
    
    print(f"\n ESTACIONES ÉXITO:")
    print(f"   • Cantidad: {insights['estaciones_exito_count']}")
    print(f"   • % de transacciones: {insights['pct_transacciones_exito']}")
    print(f"   • % de ingresos: {insights['pct_ingresos_exito']}")
    print(f"   • Ingresos: ${insights['ingresos_exito']:,} COP")
    
    print(f"\n  BAJO RENDIMIENTO (<50 transacciones):")
    print(f"   • Cantidad: {insights['estaciones_bajo_rendimiento']}")
    print(f"   • Porcentaje: {insights['pct_bajo_rendimiento']}")
    
    print(f"\n INSIGHTS PRINCIPALES:")
    print(f"\n   1. CONCENTRACIÓN:")
    print(f"      {insights['insight_principal']}")
    
    print(f"\n   2. MODELO ÉXITO:")
    print(f"      {insights['insight_exito']}")
    
    print(f"\n   3. OPORTUNIDAD DE OPTIMIZACIÓN:")
    print(f"      {insights['insight_bajo_rendimiento']}")
    
    # Guardar gráfico
    print("\n" + "="*80)
    print(" Guardando gráfico...")
    guardar_grafico(fig, 'grafico_01_barras_estaciones.png')
    
    # Mostrar en navegador
    print(" Abriendo gráfico en navegador...")
    fig.show()
    
    print("\n" + "="*80)
    print(" GRÁFICO 1 COMPLETADO")
    print("="*80)
    print(f"\nUsando {len(df):,} registros con datos normalizados")
    print("Archivo guardado: outputs/grafico_01_barras_estaciones.png")
    print("="*80 + "\n")