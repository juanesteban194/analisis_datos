"""
Gráfico 3: Segmentación RFM
Análisis: Recency, Frequency, Monetary (replica del análisis Colab)
Tipo: Gráfico de dispersión 3D interactivo
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import os
import sys
from datetime import datetime

# Agregar el directorio raíz al path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from utils import cargar_datos, guardar_grafico, formatear_moneda

def calcular_rfm(df):
    """
    Calcula métricas RFM para cada usuario
    
    Args:
        df: DataFrame con datos de Oasis
    
    Returns:
        DataFrame con métricas RFM por usuario
    """
    
    # Fecha de referencia (última fecha en el dataset)
    fecha_maxima = df['start_date_time'].max()
    
    # Calcular métricas por usuario
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
    
    # Calcular scores RFM (1-5, donde 5 es mejor)
    # Usar qcut con manejo de duplicados más robusto
    try:
        rfm['r_score'] = pd.qcut(rfm['recency'], q=5, labels=False, duplicates='drop') + 1
        # Invertir para que menos días = mejor score
        rfm['r_score'] = 6 - rfm['r_score']
    except ValueError:
        # Si falla qcut, usar percentiles manualmente
        rfm['r_score'] = pd.cut(rfm['recency'], 
                                bins=5, 
                                labels=[5, 4, 3, 2, 1],
                                duplicates='drop')
        rfm['r_score'] = rfm['r_score'].astype(int)
    
    try:
        rfm['f_score'] = pd.qcut(rfm['frequency'], q=5, labels=False, duplicates='drop') + 1
    except ValueError:
        rfm['f_score'] = pd.cut(rfm['frequency'], 
                                bins=5, 
                                labels=[1, 2, 3, 4, 5],
                                duplicates='drop')
        rfm['f_score'] = rfm['f_score'].astype(int)
    
    try:
        rfm['m_score'] = pd.qcut(rfm['monetary'], q=5, labels=False, duplicates='drop') + 1
    except ValueError:
        rfm['m_score'] = pd.cut(rfm['monetary'], 
                                bins=5, 
                                labels=[1, 2, 3, 4, 5],
                                duplicates='drop')
        rfm['m_score'] = rfm['m_score'].astype(int)
    
    # Score RFM combinado
    rfm['rfm_score'] = rfm['r_score'] + rfm['f_score'] + rfm['m_score']
    
    # Segmentación basada en RFM score
    def segmentar_usuario(row):
        rfm_score = row['rfm_score']
        r_score = row['r_score']
        f_score = row['f_score']
        m_score = row['m_score']
        
        # Champions: Compraron recientemente, compran frecuente y gastan mucho
        if rfm_score >= 13:
            return 'Champions'
        # Loyal: Alta frecuencia
        elif f_score >= 4:
            return 'Loyal'
        # Potential Loyalist: Reciente y frecuente
        elif r_score >= 4 and f_score >= 3:
            return 'Potential Loyalist'
        # At Risk: Buenos clientes pero no han comprado recientemente
        elif r_score <= 2 and f_score >= 3:
            return 'At Risk'
        # Can't Lose: Gastaron mucho pero hace tiempo
        elif r_score <= 2 and m_score >= 4:
            return "Can't Lose"
        # Hibernating: No han comprado hace mucho
        elif r_score <= 2:
            return 'Hibernating'
        # Promising: Nuevos clientes con potencial
        elif r_score >= 4 and f_score <= 2:
            return 'Promising'
        else:
            return 'Need Attention'
    
    rfm['segment'] = rfm.apply(segmentar_usuario, axis=1)
    
    return rfm

def crear_grafico(rfm_df):
    """
    Crea gráfico de dispersión 3D con segmentación RFM
    
    Args:
        rfm_df: DataFrame con métricas RFM
    
    Returns:
        Figure de Plotly
    """
    
    # Colores por segmento
    color_map = {
        'Champions': '#10b981',  # Verde
        'Loyal': '#3b82f6',  # Azul
        'Potential Loyalist': '#8b5cf6',  # Morado
        'At Risk': '#ef4444',  # Rojo
        "Can't Lose": '#dc2626',  # Rojo oscuro
        'Hibernating': '#6b7280',  # Gris
        'Promising': '#f59e0b',  # Amarillo
        'Need Attention': '#f97316'  # Naranja
    }
    
    # Crear gráfico 3D
    fig = go.Figure()
    
    for segment in sorted(rfm_df['segment'].unique()):
        df_segment = rfm_df[rfm_df['segment'] == segment]
        
        fig.add_trace(go.Scatter3d(
            x=df_segment['recency'],
            y=df_segment['frequency'],
            z=df_segment['monetary'],
            mode='markers',
            name=segment,
            marker=dict(
                size=6,
                color=color_map.get(segment, '#64748b'),
                opacity=0.7,
                line=dict(width=0.5, color='white')
            ),
            text=df_segment['user_id'],
            hovertemplate=(
                '<b>Segmento: %{fullData.name}</b><br>' +
                'User ID: %{text}<br>' +
                'Recency: %{x} días<br>' +
                'Frequency: %{y} transacciones<br>' +
                'Monetary: $%{z:,.0f} COP<br>' +
                '<extra></extra>'
            )
        ))
    
    fig.update_layout(
        title={
            'text': 'Segmentación RFM de Usuarios - Vista 3D Interactiva',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#2d3748', 'family': 'Arial Black'}
        },
        scene=dict(
            xaxis=dict(
                title='Recency (días desde última compra)',
                backgroundcolor='rgb(230, 230,230)',
                gridcolor='white'
            ),
            yaxis=dict(
                title='Frequency (número de transacciones)',
                backgroundcolor='rgb(230, 230,230)',
                gridcolor='white'
            ),
            zaxis=dict(
                title='Monetary (gasto total COP)',
                backgroundcolor='rgb(230, 230,230)',
                gridcolor='white'
            )
        ),
        height=800,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        font=dict(family='Arial', size=12)
    )
    
    return fig

def crear_grafico_2d(rfm_df):
    """
    Crea gráfico 2D alternativo (Frequency vs Monetary)
    
    Args:
        rfm_df: DataFrame con métricas RFM
    
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
    
    fig = px.scatter(
        rfm_df,
        x='frequency',
        y='monetary',
        color='segment',
        size='recency',
        size_max=20,
        color_discrete_map=color_map,
        hover_data=['user_id', 'recency', 'rfm_score'],
        labels={
            'frequency': 'Frequency (Transacciones)',
            'monetary': 'Monetary (Gasto Total COP)',
            'segment': 'Segmento'
        },
        title='Segmentación RFM - Frequency vs Monetary (tamaño = Recency)'
    )
    
    fig.update_layout(
        title={
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#2d3748', 'family': 'Arial Black'}
        },
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=700,
        font=dict(family='Arial', size=12)
    )
    
    fig.update_xaxes(gridcolor='#e2e8f0')
    fig.update_yaxes(gridcolor='#e2e8f0')
    
    return fig

def analizar(rfm_df):
    """
    Analiza la segmentación RFM y retorna insights
    
    Args:
        rfm_df: DataFrame con métricas RFM
    
    Returns:
        dict con métricas e insights
    """
    
    # Estadísticas por segmento
    segment_stats = rfm_df.groupby('segment').agg({
        'user_id': 'count',
        'recency': 'mean',
        'frequency': 'mean',
        'monetary': 'mean',
        'rfm_score': 'mean'
    }).round(2)
    
    segment_stats.columns = ['usuarios', 'recency_avg', 'frequency_avg', 'monetary_avg', 'rfm_score_avg']
    segment_stats = segment_stats.sort_values('usuarios', ascending=False)
    
    # Calcular porcentajes
    total_usuarios = len(rfm_df)
    segment_stats['porcentaje'] = (segment_stats['usuarios'] / total_usuarios * 100).round(1)
    
    # Valor total por segmento
    segment_value = rfm_df.groupby('segment')['monetary'].sum().sort_values(ascending=False)
    
    # Identificar segmentos clave
    top_segment = segment_stats.index[0]
    top_segment_pct = segment_stats.loc[top_segment, 'porcentaje']
    
    champions = rfm_df[rfm_df['segment'] == 'Champions']
    at_risk = rfm_df[rfm_df['segment'] == 'At Risk']
    
    insights = {
        'total_usuarios': total_usuarios,
        'total_segmentos': len(segment_stats),
        'segment_stats': segment_stats,
        'segment_value': segment_value,
        
        'champions_count': len(champions),
        'champions_pct': f"{(len(champions)/total_usuarios*100):.1f}%",
        'champions_value': int(champions['monetary'].sum()) if len(champions) > 0 else 0,
        
        'at_risk_count': len(at_risk),
        'at_risk_pct': f"{(len(at_risk)/total_usuarios*100):.1f}%",
        'at_risk_value': int(at_risk['monetary'].sum()) if len(at_risk) > 0 else 0,
        
        'top_segment': top_segment,
        'top_segment_pct': f"{top_segment_pct:.1f}%",
        
        'insight_champions': f"Los {len(champions)} usuarios Champions ({(len(champions)/total_usuarios*100):.1f}%) generan ${champions['monetary'].sum():,.0f} COP en ingresos. Son los mejores clientes." if len(champions) > 0 else "No hay usuarios Champions actualmente.",
        
        'insight_at_risk': f"Hay {len(at_risk)} usuarios At Risk ({(len(at_risk)/total_usuarios*100):.1f}%) que representan ${at_risk['monetary'].sum():,.0f} COP. Requieren atención urgente antes de perderlos." if len(at_risk) > 0 else "No hay usuarios At Risk.",
        
        'insight_distribucion': f"El segmento más grande es '{top_segment}' con {top_segment_pct}% de usuarios. La estrategia debe diferenciarse por segmento."
    }
    
    return insights

if __name__ == "__main__":
    print("\n" + "="*80)
    print("GRÁFICO 3: SEGMENTACIÓN RFM")
    print("="*80)
    
    # Cargar datos
    csv_path = os.path.join(parent_dir, 'data', 'df_oasis_clean.csv')
    df = cargar_datos(csv_path)
    print(f"✓ Datos cargados: {len(df):,} registros\n")
    
    # Calcular RFM
    print("🔄 Calculando métricas RFM...")
    rfm_df = calcular_rfm(df)
    print(f"✓ RFM calculado para {len(rfm_df):,} usuarios\n")
    
    # Crear gráfico 3D
    print("📊 Creando gráfico 3D...")
    fig_3d = crear_grafico(rfm_df)
    
    # Crear gráfico 2D
    print("📊 Creando gráfico 2D...")
    fig_2d = crear_grafico_2d(rfm_df)
    
    # Analizar datos
    print("\n" + "="*80)
    print("📊 ANÁLISIS E INSIGHTS")
    print("="*80)
    
    insights = analizar(rfm_df)
    
    print(f"\n📈 RESUMEN GENERAL:")
    print(f"   • Total de usuarios: {insights['total_usuarios']:,}")
    print(f"   • Segmentos identificados: {insights['total_segmentos']}")
    
    print(f"\n👑 SEGMENTO CHAMPIONS:")
    print(f"   • Usuarios: {insights['champions_count']} ({insights['champions_pct']})")
    print(f"   • Valor generado: ${insights['champions_value']:,} COP")
    
    print(f"\n⚠️  SEGMENTO AT RISK:")
    print(f"   • Usuarios: {insights['at_risk_count']} ({insights['at_risk_pct']})")
    print(f"   • Valor en riesgo: ${insights['at_risk_value']:,} COP")
    
    print(f"\n📊 DISTRIBUCIÓN POR SEGMENTO:")
    print(f"\n{'Segmento':<25} {'Usuarios':>10} {'%':>8} {'Freq Avg':>10} {'$ Avg':>15}")
    print("-" * 80)
    for segment, row in insights['segment_stats'].iterrows():
        print(f"{segment:<25} {int(row['usuarios']):>10} {row['porcentaje']:>7.1f}% {row['frequency_avg']:>10.1f} ${row['monetary_avg']:>14,.0f}")
    
    print(f"\n💰 VALOR TOTAL POR SEGMENTO (Top 5):")
    for i, (segment, value) in enumerate(insights['segment_value'].head(5).items(), 1):
        print(f"   {i}. {segment:<25} ${value:>15,.0f} COP")
    
    print(f"\n💡 INSIGHTS PRINCIPALES:")
    print(f"\n   1. CHAMPIONS - MANTENER Y RECOMPENSAR:")
    print(f"      {insights['insight_champions']}")
    
    print(f"\n   2. AT RISK - ACCIÓN URGENTE:")
    print(f"      {insights['insight_at_risk']}")
    
    print(f"\n   3. DISTRIBUCIÓN:")
    print(f"      {insights['insight_distribucion']}")
    
    print(f"\n   4. ESTRATEGIAS POR SEGMENTO:")
    print(f"      • Champions: Programas VIP, beneficios exclusivos")
    print(f"      • Loyal: Mantener engagement, cross-selling")
    print(f"      • At Risk: Campañas de re-engagement, ofertas especiales")
    print(f"      • Promising: Onboarding mejorado, incentivos iniciales")
    print(f"      • Hibernating: Win-back campaigns")
    
    # Guardar gráficos
    print("\n" + "="*80)
    print("💾 Guardando gráficos...")
    guardar_grafico(fig_3d, 'grafico_03_rfm_3d.png')
    guardar_grafico(fig_2d, 'grafico_03_rfm_2d.png')
    
    # Mostrar en navegador
    print("🌐 Abriendo gráfico 3D en navegador...")
    fig_3d.show()
    
    print("\n🌐 Abriendo gráfico 2D en navegador...")
    fig_2d.show()
    
    print("\n" + "="*80)
    print("✅ GRÁFICO 3 COMPLETADO")
    print("="*80)
    print("\nArchivos guardados:")
    print("  • outputs/grafico_03_rfm_3d.png")
    print("  • outputs/grafico_03_rfm_2d.png")
    print("\n💡 Próximo paso: Implementar grafico_04_clv_segmentos.py")
    print("="*80 + "\n")