"""
Gráfico 2: Distribución de Usuarios
Análisis: Histograma de transacciones por usuario (replica del análisis Colab)
Tipo: Histograma con curva de densidad
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
import sys

# Agregar el directorio raíz al path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from utils import cargar_datos, guardar_grafico, formatear_numero

def crear_grafico(df):
    """
    Crea el histograma de distribución de transacciones por usuario
    
    Args:
        df: DataFrame con datos de Oasis
    
    Returns:
        Figure de Plotly
    """
    
    # Contar transacciones por usuario
    user_transactions = df.groupby('user_id').size()
    
    # Crear figura con subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Histograma de Transacciones por Usuario', 
                       'Curva de Densidad (KDE)'),
        vertical_spacing=0.12,
        row_heights=[0.6, 0.4]
    )
    
    # Histograma
    fig.add_trace(
        go.Histogram(
            x=user_transactions,
            nbinsx=50,
            name='Usuarios',
            marker=dict(
                color='rgba(102, 126, 234, 0.7)',
                line=dict(color='rgba(102, 126, 234, 1)', width=1)
            ),
            hovertemplate='Transacciones: %{x}<br>Usuarios: %{y}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Calcular KDE manualmente
    from scipy import stats
    kde = stats.gaussian_kde(user_transactions)
    x_range = np.linspace(user_transactions.min(), user_transactions.max(), 200)
    density = kde(x_range)
    
    # Curva de densidad
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=density,
            mode='lines',
            name='Densidad',
            line=dict(color='rgba(118, 75, 162, 0.8)', width=3),
            fill='tozeroy',
            fillcolor='rgba(118, 75, 162, 0.2)',
            hovertemplate='Transacciones: %{x:.1f}<br>Densidad: %{y:.4f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Layout
    fig.update_layout(
        title={
            'text': 'Distribución de Transacciones por Usuario',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2d3748', 'family': 'Arial Black'}
        },
        showlegend=True,
        height=800,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Arial', size=12),
        hovermode='x unified'
    )
    
    # Actualizar ejes X
    fig.update_xaxes(
        title_text="Número de Transacciones por Usuario",
        gridcolor='#e2e8f0',
        row=1, col=1
    )
    fig.update_xaxes(
        title_text="Número de Transacciones",
        gridcolor='#e2e8f0',
        row=2, col=1
    )
    
    # Actualizar ejes Y
    fig.update_yaxes(
        title_text="Número de Usuarios",
        gridcolor='#e2e8f0',
        row=1, col=1
    )
    fig.update_yaxes(
        title_text="Densidad",
        gridcolor='#e2e8f0',
        row=2, col=1
    )
    
    return fig

def analizar(df):
    """
    Analiza la distribución de usuarios y retorna insights
    
    Args:
        df: DataFrame con datos
    
    Returns:
        dict con métricas e insights
    """
    
    # Contar transacciones por usuario
    user_transactions = df.groupby('user_id').size()
    
    # Estadísticas descriptivas
    total_usuarios = len(user_transactions)
    promedio = user_transactions.mean()
    mediana = user_transactions.median()
    desv_std = user_transactions.std()
    
    # Percentiles
    p25 = user_transactions.quantile(0.25)
    p75 = user_transactions.quantile(0.75)
    p90 = user_transactions.quantile(0.90)
    p95 = user_transactions.quantile(0.95)
    
    # Mínimo y máximo
    min_trans = user_transactions.min()
    max_trans = user_transactions.max()
    
    # Categorización de usuarios
    usuarios_ocasionales = len(user_transactions[user_transactions <= 3])
    usuarios_regulares = len(user_transactions[(user_transactions > 3) & (user_transactions <= 10)])
    usuarios_frecuentes = len(user_transactions[(user_transactions > 10) & (user_transactions <= 20)])
    usuarios_vip = len(user_transactions[user_transactions > 20])
    
    # Porcentajes
    pct_ocasionales = (usuarios_ocasionales / total_usuarios * 100)
    pct_regulares = (usuarios_regulares / total_usuarios * 100)
    pct_frecuentes = (usuarios_frecuentes / total_usuarios * 100)
    pct_vip = (usuarios_vip / total_usuarios * 100)
    
    # Concentración (20% de usuarios que más usan)
    top_20_pct = int(total_usuarios * 0.2)
    top_20_usuarios = user_transactions.nlargest(top_20_pct)
    transacciones_top_20 = top_20_usuarios.sum()
    total_transacciones = user_transactions.sum()
    concentracion_top_20 = (transacciones_top_20 / total_transacciones * 100)
    
    insights = {
        'total_usuarios': total_usuarios,
        'total_transacciones': int(total_transacciones),
        'promedio': f"{promedio:.2f}",
        'mediana': int(mediana),
        'desviacion_std': f"{desv_std:.2f}",
        
        'min_transacciones': int(min_trans),
        'max_transacciones': int(max_trans),
        
        'percentil_25': int(p25),
        'percentil_75': int(p75),
        'percentil_90': int(p90),
        'percentil_95': int(p95),
        
        'usuarios_ocasionales': usuarios_ocasionales,
        'pct_ocasionales': f"{pct_ocasionales:.1f}%",
        
        'usuarios_regulares': usuarios_regulares,
        'pct_regulares': f"{pct_regulares:.1f}%",
        
        'usuarios_frecuentes': usuarios_frecuentes,
        'pct_frecuentes': f"{pct_frecuentes:.1f}%",
        
        'usuarios_vip': usuarios_vip,
        'pct_vip': f"{pct_vip:.1f}%",
        
        'concentracion_top_20': f"{concentracion_top_20:.1f}%",
        
        'insight_principal': f"La mayoría de usuarios ({pct_ocasionales:.1f}%) son ocasionales (≤3 transacciones), mientras que solo {pct_vip:.1f}% son usuarios VIP (>20 transacciones).",
        
        'insight_concentracion': f"El 20% de usuarios más activos concentran el {concentracion_top_20:.1f}% de todas las transacciones, mostrando una distribución muy desigual.",
        
        'insight_mediana': f"La mediana de {int(mediana)} transacciones es mucho menor que el promedio de {promedio:.1f}, indicando una distribución sesgada hacia la derecha con usuarios muy activos que elevan el promedio."
    }
    
    return insights

if __name__ == "__main__":
    print("\n" + "="*80)
    print("GRÁFICO 2: DISTRIBUCIÓN DE USUARIOS")
    print("="*80)
    
    # Cargar datos
    csv_path = os.path.join(parent_dir, 'data', 'df_oasis_clean.csv')
    df = cargar_datos(csv_path)
    print(f"✓ Datos cargados: {len(df):,} registros\n")
    
    # Crear gráfico
    print("📊 Creando gráfico...")
    fig = crear_grafico(df)
    
    # Analizar datos
    print("\n" + "="*80)
    print("📊 ANÁLISIS E INSIGHTS")
    print("="*80)
    
    insights = analizar(df)
    
    print(f"\n📈 ESTADÍSTICAS DESCRIPTIVAS:")
    print(f"   • Total de usuarios: {insights['total_usuarios']:,}")
    print(f"   • Total de transacciones: {insights['total_transacciones']:,}")
    print(f"   • Promedio por usuario: {insights['promedio']}")
    print(f"   • Mediana: {insights['mediana']}")
    print(f"   • Desviación estándar: {insights['desviacion_std']}")
    print(f"   • Rango: {insights['min_transacciones']} - {insights['max_transacciones']}")
    
    print(f"\n📊 PERCENTILES:")
    print(f"   • P25 (25%): {insights['percentil_25']} transacciones")
    print(f"   • P50 (mediana): {insights['mediana']} transacciones")
    print(f"   • P75 (75%): {insights['percentil_75']} transacciones")
    print(f"   • P90 (90%): {insights['percentil_90']} transacciones")
    print(f"   • P95 (95%): {insights['percentil_95']} transacciones")
    
    print(f"\n👥 SEGMENTACIÓN DE USUARIOS:")
    print(f"   • Ocasionales (≤3 trans):   {insights['usuarios_ocasionales']:>4} usuarios ({insights['pct_ocasionales']})")
    print(f"   • Regulares (4-10 trans):    {insights['usuarios_regulares']:>4} usuarios ({insights['pct_regulares']})")
    print(f"   • Frecuentes (11-20 trans):  {insights['usuarios_frecuentes']:>4} usuarios ({insights['pct_frecuentes']})")
    print(f"   • VIP (>20 trans):           {insights['usuarios_vip']:>4} usuarios ({insights['pct_vip']})")
    
    print(f"\n🎯 CONCENTRACIÓN:")
    print(f"   • Top 20% de usuarios concentran: {insights['concentracion_top_20']}")
    
    print(f"\n💡 INSIGHTS PRINCIPALES:")
    print(f"\n   1. MAYORÍA OCASIONAL:")
    print(f"      {insights['insight_principal']}")
    
    print(f"\n   2. CONCENTRACIÓN EXTREMA:")
    print(f"      {insights['insight_concentracion']}")
    
    print(f"\n   3. DISTRIBUCIÓN SESGADA:")
    print(f"      {insights['insight_mediana']}")
    
    print(f"\n   4. OPORTUNIDAD:")
    print(f"      Convertir usuarios ocasionales en regulares podría duplicar las")
    print(f"      transacciones. Programas de fidelización y beneficios por frecuencia")
    print(f"      son estrategias clave.")
    
    # Guardar gráfico
    print("\n" + "="*80)
    print("💾 Guardando gráfico...")
    guardar_grafico(fig, 'grafico_02_distribucion_usuarios.png')
    
    # Mostrar en navegador
    print("🌐 Abriendo gráfico en navegador...")
    fig.show()
    
    