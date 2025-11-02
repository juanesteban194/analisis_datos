"""
Gráfico 4: CLV por Segmentos
Análisis: Customer Lifetime Value (Valor del Tiempo de Vida del Cliente) por segmentos RFM
Tipo: Gráfico de barras con métricas comparativas
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

from utils import cargar_datos, guardar_grafico, formatear_moneda

def calcular_rfm(df):
    """
    Calcula métricas RFM (reutilizado del gráfico 3)
    """
    fecha_maxima = df['start_date_time'].max()
    
    rfm_data = []
    for user_id, group in df.groupby('user_id'):
        recency = (fecha_maxima - group['start_date_time'].max()).days
        frequency = len(group)
        monetary = group['amount_transaction'].sum()
        
        rfm_data.append({
            'user_id': user_id,
            'recency': recency,
            'frequency': frequency,
            'monetary': monetary
        })
    
    rfm = pd.DataFrame(rfm_data)
    
    # Calcular scores
    try:
        rfm['r_score'] = pd.qcut(rfm['recency'], q=5, labels=False, duplicates='drop') + 1
        rfm['r_score'] = 6 - rfm['r_score']
    except ValueError:
        rfm['r_score'] = pd.cut(rfm['recency'], bins=5, labels=[5, 4, 3, 2, 1], duplicates='drop')
        rfm['r_score'] = rfm['r_score'].astype(int)
    
    try:
        rfm['f_score'] = pd.qcut(rfm['frequency'], q=5, labels=False, duplicates='drop') + 1
    except ValueError:
        rfm['f_score'] = pd.cut(rfm['frequency'], bins=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        rfm['f_score'] = rfm['f_score'].astype(int)
    
    try:
        rfm['m_score'] = pd.qcut(rfm['monetary'], q=5, labels=False, duplicates='drop') + 1
    except ValueError:
        rfm['m_score'] = pd.cut(rfm['monetary'], bins=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        rfm['m_score'] = rfm['m_score'].astype(int)
    
    rfm['rfm_score'] = rfm['r_score'] + rfm['f_score'] + rfm['m_score']
    
    def segmentar_usuario(row):
        rfm_score = row['rfm_score']
        r_score = row['r_score']
        f_score = row['f_score']
        m_score = row['m_score']
        
        if rfm_score >= 13:
            return 'Champions'
        elif f_score >= 4:
            return 'Loyal'
        elif r_score >= 4 and f_score >= 3:
            return 'Potential Loyalist'
        elif r_score <= 2 and f_score >= 3:
            return 'At Risk'
        elif r_score <= 2 and m_score >= 4:
            return "Can't Lose"
        elif r_score <= 2:
            return 'Hibernating'
        elif r_score >= 4 and f_score <= 2:
            return 'Promising'
        else:
            return 'Need Attention'
    
    rfm['segment'] = rfm.apply(segmentar_usuario, axis=1)
    
    return rfm

def calcular_clv(df, rfm_df):
    """
    Calcula el Customer Lifetime Value por segmento
    
    Args:
        df: DataFrame original con transacciones
        rfm_df: DataFrame con segmentación RFM
    
    Returns:
        DataFrame con métricas CLV por segmento
    """
    
    # Calcular duración del cliente (días desde primera a última compra)
    user_lifetime = df.groupby('user_id').agg({
        'start_date_time': lambda x: (x.max() - x.min()).days + 1
    }).reset_index()
    user_lifetime.columns = ['user_id', 'lifetime_days']
    
    # Merge con RFM
    rfm_with_lifetime = rfm_df.merge(user_lifetime, on='user_id')
    
    # Calcular CLV y métricas por segmento
    clv_by_segment = rfm_with_lifetime.groupby('segment').agg({
        'user_id': 'count',
        'monetary': ['sum', 'mean'],
        'frequency': 'mean',
        'lifetime_days': 'mean',
        'recency': 'mean'
    }).reset_index()
    
    # Aplanar columnas multi-nivel
    clv_by_segment.columns = ['segment', 'num_users', 'total_revenue', 
                                'avg_revenue_per_user', 'avg_frequency', 
                                'avg_lifetime_days', 'avg_recency']
    
    # Calcular CLV (simplificado)
    # CLV = Valor promedio por transacción * Frecuencia * Tiempo de vida esperado
    clv_by_segment['avg_transaction_value'] = (
        clv_by_segment['avg_revenue_per_user'] / clv_by_segment['avg_frequency']
    )
    
    # Proyectar CLV a 1 año (365 días)
    clv_by_segment['transactions_per_year'] = (
        365 / (clv_by_segment['avg_lifetime_days'] / clv_by_segment['avg_frequency'])
    )
    
    clv_by_segment['clv_projected_1year'] = (
        clv_by_segment['avg_transaction_value'] * clv_by_segment['transactions_per_year']
    )
    
    # Calcular porcentaje de usuarios
    total_users = clv_by_segment['num_users'].sum()
    clv_by_segment['user_percentage'] = (clv_by_segment['num_users'] / total_users * 100)
    
    # Calcular porcentaje de ingresos
    total_revenue = clv_by_segment['total_revenue'].sum()
    clv_by_segment['revenue_percentage'] = (clv_by_segment['total_revenue'] / total_revenue * 100)
    
    # Ordenar por CLV proyectado
    clv_by_segment = clv_by_segment.sort_values('clv_projected_1year', ascending=False)
    
    return clv_by_segment

def crear_grafico(clv_df):
    """
    Crea gráfico de barras con CLV por segmento
    
    Args:
        clv_df: DataFrame con métricas CLV
    
    Returns:
        Figure de Plotly
    """
    
    # Colores por segmento
    color_map = {
        'Champions': '#10b981',
        'Loyal': '#3b82f6',
        'Potential Loyalist': '#8b5cf6',
        'At Risk': '#ef4444',
        "Can't Lose": '#dc2626',
        'Hibernating': '#6b7280',
        'Promising': '#f59e0b',
        'Need Attention': '#f97316'
    }
    
    colors = [color_map.get(seg, '#64748b') for seg in clv_df['segment']]
    
    # Crear subplots: 2 filas
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'CLV Proyectado a 1 Año por Segmento',
            'Ingresos Totales por Segmento',
            'Número de Usuarios por Segmento',
            'Valor Promedio por Transacción'
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )
    
    # Gráfico 1: CLV Proyectado
    fig.add_trace(
        go.Bar(
            x=clv_df['segment'],
            y=clv_df['clv_projected_1year'],
            marker_color=colors,
            text=[f"${val:,.0f}" for val in clv_df['clv_projected_1year']],
            textposition='outside',
            name='CLV',
            hovertemplate='<b>%{x}</b><br>CLV: $%{y:,.0f} COP<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Gráfico 2: Ingresos Totales
    fig.add_trace(
        go.Bar(
            x=clv_df['segment'],
            y=clv_df['total_revenue'],
            marker_color=colors,
            text=[f"${val/1e6:.1f}M" for val in clv_df['total_revenue']],
            textposition='outside',
            name='Ingresos',
            hovertemplate='<b>%{x}</b><br>Ingresos: $%{y:,.0f} COP<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Gráfico 3: Número de Usuarios
    fig.add_trace(
        go.Bar(
            x=clv_df['segment'],
            y=clv_df['num_users'],
            marker_color=colors,
            text=[f"{val}<br>({pct:.1f}%)" for val, pct in 
                  zip(clv_df['num_users'], clv_df['user_percentage'])],
            textposition='outside',
            name='Usuarios',
            hovertemplate='<b>%{x}</b><br>Usuarios: %{y}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Gráfico 4: Valor Promedio por Transacción
    fig.add_trace(
        go.Bar(
            x=clv_df['segment'],
            y=clv_df['avg_transaction_value'],
            marker_color=colors,
            text=[f"${val:,.0f}" for val in clv_df['avg_transaction_value']],
            textposition='outside',
            name='Valor/Trans',
            hovertemplate='<b>%{x}</b><br>Valor promedio: $%{y:,.0f} COP<extra></extra>'
        ),
        row=2, col=2
    )
    
    # Layout
    fig.update_layout(
        title={
            'text': 'Análisis de Customer Lifetime Value (CLV) por Segmento',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2d3748', 'family': 'Arial Black'}
        },
        showlegend=False,
        height=900,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Arial', size=11)
    )
    
    # Actualizar ejes
    fig.update_xaxes(tickangle=-45, tickfont={'size': 9})
    fig.update_yaxes(gridcolor='#e2e8f0')
    
    # Títulos de ejes Y
    fig.update_yaxes(title_text="CLV (COP)", row=1, col=1)
    fig.update_yaxes(title_text="Ingresos Totales (COP)", row=1, col=2)
    fig.update_yaxes(title_text="Número de Usuarios", row=2, col=1)
    fig.update_yaxes(title_text="Valor por Transacción (COP)", row=2, col=2)
    
    return fig

def analizar(clv_df):
    """
    Analiza métricas CLV y genera insights
    
    Args:
        clv_df: DataFrame con métricas CLV
    
    Returns:
        dict con insights
    """
    
    # Identificar segmento con mayor CLV
    top_clv_segment = clv_df.iloc[0]
    
    # Calcular totales
    total_users = clv_df['num_users'].sum()
    total_revenue = clv_df['total_revenue'].sum()
    
    # Champions vs resto
    champions = clv_df[clv_df['segment'] == 'Champions']
    if len(champions) > 0:
        champions_clv = champions.iloc[0]['clv_projected_1year']
        champions_users = champions.iloc[0]['num_users']
        champions_revenue = champions.iloc[0]['total_revenue']
    else:
        champions_clv = 0
        champions_users = 0
        champions_revenue = 0
    
    # At Risk
    at_risk = clv_df[clv_df['segment'] == 'At Risk']
    if len(at_risk) > 0:
        at_risk_revenue = at_risk.iloc[0]['total_revenue']
        at_risk_users = at_risk.iloc[0]['num_users']
    else:
        at_risk_revenue = 0
        at_risk_users = 0
    
    # ROI potencial
    roi_recuperacion = at_risk_revenue * 0.3  # Estimar 30% de recuperación
    
    insights = {
        'total_users': int(total_users),
        'total_revenue': int(total_revenue),
        
        'top_clv_segment': top_clv_segment['segment'],
        'top_clv_value': top_clv_segment['clv_projected_1year'],
        'top_clv_users': int(top_clv_segment['num_users']),
        'top_clv_revenue': int(top_clv_segment['total_revenue']),
        
        'champions_clv': champions_clv,
        'champions_users': int(champions_users),
        'champions_revenue': int(champions_revenue),
        'champions_pct_revenue': (champions_revenue / total_revenue * 100) if champions_revenue > 0 else 0,
        
        'at_risk_users': int(at_risk_users),
        'at_risk_revenue': int(at_risk_revenue),
        'at_risk_pct': (at_risk_users / total_users * 100) if at_risk_users > 0 else 0,
        
        'roi_recuperacion': int(roi_recuperacion),
        
        'clv_table': clv_df,
        
        'insight_top_clv': f"El segmento '{top_clv_segment['segment']}' tiene el mayor CLV proyectado (${top_clv_segment['clv_projected_1year']:,.0f} COP/año), con {int(top_clv_segment['num_users'])} usuarios generando ${top_clv_segment['total_revenue']:,.0f} COP.",
        
        'insight_champions': f"Los Champions tienen un CLV de ${champions_clv:,.0f} COP/año y representan el {(champions_revenue / total_revenue * 100):.1f}% de los ingresos totales con solo {champions_users} usuarios." if champions_users > 0 else "No hay usuarios Champions actualmente.",
        
        'insight_at_risk': f"Los {at_risk_users} usuarios At Risk representan ${at_risk_revenue:,.0f} COP en riesgo. Recuperar el 30% generaría ${roi_recuperacion:,.0f} COP adicionales." if at_risk_users > 0 else "No hay usuarios At Risk.",
        
        'insight_estrategia': f"Invertir en retención de Champions (alto CLV) y recuperación de At Risk (ROI rápido) maximizará el valor a largo plazo."
    }
    
    return insights

if __name__ == "__main__":
    print("\n" + "="*80)
    print("GRÁFICO 4: CLV POR SEGMENTOS")
    print("="*80)
    
    # Cargar datos
    csv_path = os.path.join(parent_dir, 'data', 'df_oasis_clean.csv')
    df = cargar_datos(csv_path)
    print(f"✓ Datos cargados: {len(df):,} registros\n")
    
    # Calcular RFM
    print(" Calculando segmentación RFM...")
    rfm_df = calcular_rfm(df)
    print(f"✓ RFM calculado para {len(rfm_df):,} usuarios\n")
    
    # Calcular CLV
    print(" Calculando CLV por segmento...")
    clv_df = calcular_clv(df, rfm_df)
    print(f"✓ CLV calculado para {len(clv_df)} segmentos\n")
    
    # Crear gráfico
    print(" Creando gráfico...")
    fig = crear_grafico(clv_df)
    
    # Analizar
    print("\n" + "="*80)
    print("ANÁLISIS DE CLV")
    print("="*80)
    
    insights = analizar(clv_df)
    
    print(f"\nRESUMEN GENERAL:")
    print(f"   • Total de usuarios: {insights['total_users']:,}")
    print(f"   • Ingresos totales: ${insights['total_revenue']:,} COP")
    
    print(f"\nSEGMENTO CON MAYOR CLV:")
    print(f"   • Segmento: {insights['top_clv_segment']}")
    print(f"   • CLV proyectado: ${insights['top_clv_value']:,.0f} COP/año")
    print(f"   • Usuarios: {insights['top_clv_users']}")
    print(f"   • Ingresos totales: ${insights['top_clv_revenue']:,} COP")
    
    if insights['champions_users'] > 0:
        print(f"\nCHAMPIONS:")
        print(f"   • CLV: ${insights['champions_clv']:,.0f} COP/año")
        print(f"   • Usuarios: {insights['champions_users']} ({(insights['champions_users']/insights['total_users']*100):.1f}%)")
        print(f"   • Ingresos: ${insights['champions_revenue']:,} COP ({insights['champions_pct_revenue']:.1f}% del total)")
    
    if insights['at_risk_users'] > 0:
        print(f"\n AT RISK:")
        print(f"   • Usuarios: {insights['at_risk_users']} ({insights['at_risk_pct']:.1f}%)")
        print(f"   • Ingresos en riesgo: ${insights['at_risk_revenue']:,} COP")
        print(f"   • ROI potencial (30% recuperación): ${insights['roi_recuperacion']:,} COP")
    
    print(f"\nTABLA COMPLETA CLV POR SEGMENTO:")
    print(f"\n{'Segmento':<20} {'Usuarios':>8} {'CLV/año':>15} {'Ingresos Totales':>18} {'Valor/Trans':>15}")
    print("-" * 85)
    for _, row in clv_df.iterrows():
        print(f"{row['segment']:<20} {int(row['num_users']):>8} ${row['clv_projected_1year']:>14,.0f} ${row['total_revenue']:>17,.0f} ${row['avg_transaction_value']:>14,.0f}")
    
    print(f"\n INSIGHTS PRINCIPALES:")
    print(f"\n   1. MAYOR CLV:")
    print(f"      {insights['insight_top_clv']}")
    
    if insights['champions_users'] > 0:
        print(f"\n   2. CHAMPIONS:")
        print(f"      {insights['insight_champions']}")
    
    if insights['at_risk_users'] > 0:
        print(f"\n   3. OPORTUNIDAD AT RISK:")
        print(f"      {insights['insight_at_risk']}")
    
    print(f"\n   4. ESTRATEGIA:")
    print(f"      {insights['insight_estrategia']}")
    
    print(f"\n   5. PRIORIZACIÓN DE INVERSIÓN:")
    print(f"      • Alta prioridad: Champions y Loyal (máximo CLV)")
    print(f"      • Media prioridad: At Risk y Can't Lose (recuperación)")
    print(f"      • Baja prioridad: Hibernating (bajo ROI)")
    
    # Guardar gráfico
    print("\n" + "="*80)
    print(" Guardando gráfico...")
    guardar_grafico(fig, 'grafico_04_clv_segmentos.png')
    
    # Mostrar en navegador
    print("Abriendo gráfico en navegador...")
    fig.show()
    
   