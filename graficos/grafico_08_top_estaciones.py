"""
Gráfico 8: Top Estaciones - ACTUALIZADO CON DATOS NORMALIZADOS
Análisis: Análisis comparativo detallado de las mejores estaciones
- Ingresos en MILLONES COP (datos ya normalizados en el CSV)
- Total dataset: ~166.4 M COP
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

from utils import cargar_datos, guardar_grafico

def calcular_metricas_estaciones(df):
    """
    Calcula métricas agregadas por estación
    IMPORTANTE: Los datos YA están en COP (normalizados, divididos por 100)
    
    Args:
        df: DataFrame con datos de Oasis
    
    Returns:
        DataFrame con métricas por estación
    """
    
    # Agrupar por estación
    metricas = df.groupby('evse_uid').agg({
        'id': 'count',  # Total transacciones
        'user_id': 'nunique',  # Usuarios únicos
        'amount_transaction': ['sum', 'mean', 'median'],  # Ingresos (YA en COP)
        'energy_kwh': ['sum', 'mean'],  # Energía
        'start_date_time': ['min', 'max']  # Fechas
    }).reset_index()
    
    # Aplanar columnas
    metricas.columns = [
        'evse_uid', 'total_transacciones', 'usuarios_unicos',
        'ingresos_totales_cop', 'ingreso_promedio_cop', 'ingreso_mediano_cop',
        'energia_total_kwh', 'energia_promedio_kwh',
        'primera_fecha', 'ultima_fecha'
    ]
    
    # Calcular días activos
    metricas['dias_activos'] = (metricas['ultima_fecha'] - metricas['primera_fecha']).dt.days + 1
    metricas['dias_activos'] = metricas['dias_activos'].clip(lower=1)
    
    # Eficiencias
    metricas['transacciones_por_dia'] = metricas['total_transacciones'] / metricas['dias_activos']
    metricas['ingresos_por_dia_cop'] = metricas['ingresos_totales_cop'] / metricas['dias_activos']
    
    # Conversiones para visualización
    metricas['ingresos_totales_M'] = metricas['ingresos_totales_cop'] / 1_000_000  # Millones
    metricas['ingresos_por_dia_K'] = metricas['ingresos_por_dia_cop'] / 1_000  # Miles
    
    # Índice de rendimiento (0-100)
    def normalizar(serie):
        s = serie.fillna(0)
        rng = s.max() - s.min()
        if rng == 0:
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - s.min()) / rng
    
    metricas['indice_rendimiento'] = (
        normalizar(metricas['total_transacciones']) * 0.4 +
        normalizar(metricas['ingresos_totales_cop']) * 0.4 +
        normalizar(metricas['usuarios_unicos']) * 0.2
    ) * 100
    
    # Categoría por índice
    def categorizar(valor):
        if valor >= 80: return 'Elite'
        if valor >= 60: return 'Alto Rendimiento'
        if valor >= 40: return 'Rendimiento Medio'
        if valor >= 20: return 'Bajo Rendimiento'
        return 'Crítico'
    
    metricas['categoria'] = metricas['indice_rendimiento'].apply(categorizar)
    
    # Flag Éxito
    metricas['es_exito'] = metricas['evse_uid'].str.lower().str.contains('exito', na=False)
    
    # Ordenar por rendimiento
    metricas = metricas.sort_values('indice_rendimiento', ascending=False).reset_index(drop=True)
    
    return metricas

def crear_grafico_principal(metricas_df, top_n=15):
    """
    Crea el dashboard principal con 6 gráficos
    
    Args:
        metricas_df: DataFrame con métricas por estación
        top_n: Número de estaciones top a mostrar
    
    Returns:
        Figure de Plotly
    """
    
    # Colores por categoría
    color_map = {
        'Elite': '#10b981',
        'Alto Rendimiento': '#3b82f6',
        'Rendimiento Medio': '#f59e0b',
        'Bajo Rendimiento': '#ef4444',
        'Crítico': '#dc2626'
    }
    
    # Crear subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            f'Top {top_n} por Índice de Rendimiento',
            f'Top {top_n} por Ingresos Totales (Millones COP)',
            f'Top {top_n} por Transacciones',
            f'Top {top_n} por Usuarios Únicos',
            'Eficiencia: Transacciones por Día',
            'Eficiencia: Ingresos por Día (Miles COP)'
        ),
        specs=[
            [{'type': 'bar'}, {'type': 'bar'}],
            [{'type': 'bar'}, {'type': 'bar'}],
            [{'type': 'bar'}, {'type': 'bar'}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.15,
        row_heights=[0.34, 0.33, 0.33]
    )
    
    # 1) Índice de rendimiento
    top_indice = metricas_df.head(top_n)
    fig.add_trace(
        go.Bar(
            y=top_indice['evse_uid'],
            x=top_indice['indice_rendimiento'],
            orientation='h',
            marker_color=[color_map[c] for c in top_indice['categoria']],
            text=[f"{v:.1f}" for v in top_indice['indice_rendimiento']],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Índice: %{x:.1f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 2) Ingresos totales (MILLONES)
    top_ingresos = metricas_df.nlargest(top_n, 'ingresos_totales_M').sort_values('ingresos_totales_M')
    fig.add_trace(
        go.Bar(
            y=top_ingresos['evse_uid'],
            x=top_ingresos['ingresos_totales_M'],
            orientation='h',
            marker_color='#10b981',
            text=[f"${v:.1f}M" for v in top_ingresos['ingresos_totales_M']],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Ingresos: $%{x:.2f}M COP<extra></extra>'
        ),
        row=1, col=2
    )
    
    # 3) Transacciones
    top_trans = metricas_df.nlargest(top_n, 'total_transacciones').sort_values('total_transacciones')
    fig.add_trace(
        go.Bar(
            y=top_trans['evse_uid'],
            x=top_trans['total_transacciones'],
            orientation='h',
            marker_color='#3b82f6',
            text=[f"{int(v)}" for v in top_trans['total_transacciones']],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Transacciones: %{x}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # 4) Usuarios únicos
    top_usuarios = metricas_df.nlargest(top_n, 'usuarios_unicos').sort_values('usuarios_unicos')
    fig.add_trace(
        go.Bar(
            y=top_usuarios['evse_uid'],
            x=top_usuarios['usuarios_unicos'],
            orientation='h',
            marker_color='#8b5cf6',
            text=[f"{int(v)}" for v in top_usuarios['usuarios_unicos']],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Usuarios: %{x}<extra></extra>'
        ),
        row=2, col=2
    )
    
    # 5) Transacciones por día
    top_trans_dia = metricas_df.nlargest(top_n, 'transacciones_por_dia').sort_values('transacciones_por_dia')
    fig.add_trace(
        go.Bar(
            y=top_trans_dia['evse_uid'],
            x=top_trans_dia['transacciones_por_dia'],
            orientation='h',
            marker_color='#f59e0b',
            text=[f"{v:.1f}" for v in top_trans_dia['transacciones_por_dia']],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Trans/Día: %{x:.2f}<extra></extra>'
        ),
        row=3, col=1
    )
    
    # 6) Ingresos por día (MILES)
    top_ing_dia = metricas_df.nlargest(top_n, 'ingresos_por_dia_K').sort_values('ingresos_por_dia_K')
    fig.add_trace(
        go.Bar(
            y=top_ing_dia['evse_uid'],
            x=top_ing_dia['ingresos_por_dia_K'],
            orientation='h',
            marker_color='#ec4899',
            text=[f"${v:.0f}K" for v in top_ing_dia['ingresos_por_dia_K']],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>$/Día: $%{x:,.0f}K COP<extra></extra>'
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
    for r in (1, 2, 3):
        for c in (1, 2):
            fig.update_xaxes(gridcolor='#e2e8f0', row=r, col=c)
            fig.update_yaxes(tickfont=dict(size=9), row=r, col=c)
    
    fig.update_xaxes(title_text='Índice (0-100)', row=1, col=1)
    fig.update_xaxes(title_text='Ingresos Totales (Millones COP)', ticksuffix='M', row=1, col=2)
    fig.update_xaxes(title_text='Transacciones', row=2, col=1)
    fig.update_xaxes(title_text='Usuarios Únicos', row=2, col=2)
    fig.update_xaxes(title_text='Transacciones por Día', row=3, col=1)
    fig.update_xaxes(title_text='Ingresos por Día (Miles COP)', ticksuffix='K', row=3, col=2)
    
    return fig

def crear_grafico_categorias(metricas_df):
    """
    Crea gráfico de distribución por categorías
    
    Args:
        metricas_df: DataFrame con métricas
    
    Returns:
        Figure de Plotly
    """
    
    color_map = {
        'Elite': '#10b981',
        'Alto Rendimiento': '#3b82f6',
        'Rendimiento Medio': '#f59e0b',
        'Bajo Rendimiento': '#ef4444',
        'Crítico': '#dc2626'
    }
    
    orden = ['Elite', 'Alto Rendimiento', 'Rendimiento Medio', 'Bajo Rendimiento', 'Crítico']
    
    # Contar estaciones por categoría
    counts = metricas_df['categoria'].value_counts().reindex(orden, fill_value=0)
    
    # Ingresos por categoría (en millones)
    ingresos_M = (metricas_df.groupby('categoria')['ingresos_totales_cop'].sum() / 1_000_000).reindex(orden, fill_value=0)
    
    # Crear subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Distribución de Estaciones por Categoría',
                       'Ingresos por Categoría (Millones COP)')
    )
    
    # Gráfico 1: Conteo
    fig.add_trace(
        go.Bar(
            x=counts.index,
            y=counts.values,
            marker_color=[color_map[c] for c in counts.index],
            text=[f"{int(v)}" for v in counts.values],
            textposition='outside',
            hovertemplate='%{x}: %{y} estaciones<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Gráfico 2: Ingresos
    fig.add_trace(
        go.Bar(
            x=ingresos_M.index,
            y=ingresos_M.values,
            marker_color=[color_map[c] for c in ingresos_M.index],
            text=[f"${v:.1f}M" for v in ingresos_M.values],
            textposition='outside',
            hovertemplate='%{x}: $%{y:.2f}M COP<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title={
            'text': 'Análisis por Categoría de Rendimiento',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#2d3748', 'family': 'Arial Black'}
        },
        showlegend=False,
        height=520,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Arial', size=12)
    )
    
    fig.update_xaxes(gridcolor='#e2e8f0')
    fig.update_yaxes(gridcolor='#e2e8f0')
    fig.update_yaxes(title_text='Número de Estaciones', row=1, col=1)
    fig.update_yaxes(title_text='Ingresos (Millones COP)', row=1, col=2)
    
    return fig

def analizar(metricas_df, df_original, top_n=15):
    """
    Genera análisis y reconciliación de datos
    
    Args:
        metricas_df: DataFrame con métricas
        df_original: DataFrame original
        top_n: Número de estaciones top
    
    Returns:
        dict con insights
    """
    
    # Totales
    total_dataset_cop = df_original['amount_transaction'].sum()
    total_dataset_M = total_dataset_cop / 1_000_000
    
    # Verificación
    suma_estaciones_cop = metricas_df['ingresos_totales_cop'].sum()
    diferencia = abs(total_dataset_cop - suma_estaciones_cop)
    pct_diff = (diferencia / total_dataset_cop * 100) if total_dataset_cop > 0 else 0
    
    # Top N
    top = metricas_df.nlargest(top_n, 'ingresos_totales_cop')
    top_sum_M = top['ingresos_totales_cop'].sum() / 1_000_000
    pct_top = (top_sum_M / total_dataset_M * 100) if total_dataset_M > 0 else 0
    
    # Por categoría
    categoria_dist = metricas_df['categoria'].value_counts()
    
    # Éxito vs No Éxito
    exito_stats = metricas_df[metricas_df['es_exito']]
    no_exito_stats = metricas_df[~metricas_df['es_exito']]
    
    insights = {
        'total_dataset_cop': int(total_dataset_cop),
        'total_dataset_M': total_dataset_M,
        'suma_estaciones_cop': int(suma_estaciones_cop),
        'diferencia': diferencia,
        'pct_diff': pct_diff,
        'top_sum_M': top_sum_M,
        'pct_top': pct_top,
        'total_estaciones': len(metricas_df),
        
        'elite_count': int(categoria_dist.get('Elite', 0)),
        'alto_count': int(categoria_dist.get('Alto Rendimiento', 0)),
        'medio_count': int(categoria_dist.get('Rendimiento Medio', 0)),
        'bajo_count': int(categoria_dist.get('Bajo Rendimiento', 0)),
        'critico_count': int(categoria_dist.get('Crítico', 0)),
        
        'exito_count': len(exito_stats),
        'exito_ingresos_M': exito_stats['ingresos_totales_cop'].sum() / 1_000_000,
        'exito_pct': (exito_stats['ingresos_totales_cop'].sum() / total_dataset_cop * 100) if total_dataset_cop > 0 else 0,
        
        'top_1_nombre': top.iloc[0]['evse_uid'],
        'top_1_ingresos_M': top.iloc[0]['ingresos_totales_M'],
        'top_1_transacciones': int(top.iloc[0]['total_transacciones']),
        
        'top_2_nombre': top.iloc[1]['evse_uid'],
        'top_2_ingresos_M': top.iloc[1]['ingresos_totales_M'],
        
        'top_3_nombre': top.iloc[2]['evse_uid'],
        'top_3_ingresos_M': top.iloc[2]['ingresos_totales_M']
    }
    
    return insights

if __name__ == "__main__":
    print("\n" + "="*80)
    print("GRÁFICO 8: TOP ESTACIONES (DATOS NORMALIZADOS)")
    print("="*80)
    
    # Cargar datos
    csv_path = 'data/df_oasis_clean.csv'
    df = cargar_datos(csv_path)
    print(f"✓ Datos normalizados - Total: ${df['amount_transaction'].sum():,.0f} COP (~${df['amount_transaction'].sum()/1_000_000:.1f}M)\n")
    
    # Calcular métricas
    print("📊 Calculando métricas por estación...")
    metricas_df = calcular_metricas_estaciones(df)
    print(f"✓ Métricas calculadas para {len(metricas_df)} estaciones\n")
    
    # Crear gráficos
    print("🎨 Creando gráficos...")
    fig_main = crear_grafico_principal(metricas_df, top_n=15)
    fig_cat = crear_grafico_categorias(metricas_df)
    
    # Análisis
    print("\n" + "="*80)
    print("📊 ANÁLISIS Y RECONCILIACIÓN")
    print("="*80)
    
    insights = analizar(metricas_df, df, top_n=15)
    
    print(f"\n💰 RECONCILIACIÓN DE TOTALES:")
    print(f"   • Total dataset: ${insights['total_dataset_cop']:,} COP (${insights['total_dataset_M']:.2f}M)")
    print(f"   • Suma por estaciones: ${insights['suma_estaciones_cop']:,} COP")
    print(f"   • Diferencia: ${insights['diferencia']:,.2f} COP ({insights['pct_diff']:.4f}%)")
    print(f"   • Top 15 acumulan: ${insights['top_sum_M']:.2f}M ({insights['pct_top']:.1f}% del total)")
    
    print(f"\n🏆 TOP 3 ESTACIONES:")
    print(f"   1. {insights['top_1_nombre']}: ${insights['top_1_ingresos_M']:.2f}M ({insights['top_1_transacciones']:,} trans)")
    print(f"   2. {insights['top_2_nombre']}: ${insights['top_2_ingresos_M']:.2f}M")
    print(f"   3. {insights['top_3_nombre']}: ${insights['top_3_ingresos_M']:.2f}M")
    
    print(f"\n📊 DISTRIBUCIÓN POR CATEGORÍA:")
    print(f"   • Elite: {insights['elite_count']} estaciones")
    print(f"   • Alto Rendimiento: {insights['alto_count']} estaciones")
    print(f"   • Rendimiento Medio: {insights['medio_count']} estaciones")
    print(f"   • Bajo Rendimiento: {insights['bajo_count']} estaciones")
    print(f"   • Crítico: {insights['critico_count']} estaciones")
    
    print(f"\n🎯 ESTACIONES ÉXITO:")
    print(f"   • Cantidad: {insights['exito_count']}")
    print(f"   • Ingresos: ${insights['exito_ingresos_M']:.2f}M ({insights['exito_pct']:.1f}% del total)")
    
    # Guardar
    print("\n" + "="*80)
    print("💾 Guardando gráficos...")
    guardar_grafico(fig_main, 'grafico_08_top_estaciones_principal.png')
    guardar_grafico(fig_cat, 'grafico_08_categorias.png')
    
    # Mostrar
    print("🌐 Abriendo gráficos...")
    fig_main.show()
    fig_cat.show()
    
    print("\n" + "="*80)
    print("✅ GRÁFICO 8 COMPLETADO")
    print("="*80)
    print(f"\nUsando {len(df):,} registros con datos normalizados")
    print("Archivos guardados:")
    print("  • outputs/grafico_08_top_estaciones_principal.png")
    print("  • outputs/grafico_08_categorias.png")
    print("="*80 + "\n")