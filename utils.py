"""
Funciones auxiliares compartidas por todos los gráficos
"""

import pandas as pd
from pathlib import Path
import os

def cargar_datos(ruta='data/df_oasis_clean.csv'):
    """
    Carga el dataset principal de Oasis EVSE
    
    Args:
        ruta (str): Ruta al archivo CSV
    
    Returns:
        pd.DataFrame: DataFrame con los datos cargados
    """
    try:
        df = pd.read_csv(ruta)
        
        # Convertir fechas a datetime
        if 'start_date_time' in df.columns:
            df['start_date_time'] = pd.to_datetime(df['start_date_time'], format='mixed', errors='coerce')
        
        if 'end_date_time' in df.columns:
            df['end_date_time'] = pd.to_datetime(df['end_date_time'], format='mixed', errors='coerce')
        
        print(f"✓ Datos cargados: {len(df):,} registros")
        
        return df
    
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo en {ruta}")
        print(f"   Ruta completa: {os.path.abspath(ruta)}")
        raise
    except Exception as e:
        print(f"❌ Error al cargar datos: {e}")
        raise

def guardar_grafico(fig, nombre_archivo):
    """
    Guarda un gráfico como PNG en la carpeta outputs
    
    Args:
        fig: Objeto Figure de Plotly
        nombre_archivo (str): Nombre del archivo (ej: 'grafico_01.png')
    """
    try:
        # Crear carpeta outputs si no existe
        output_dir = 'outputs'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        ruta_completa = os.path.join(output_dir, nombre_archivo)
        
        # Guardar usando kaleido
        fig.write_image(ruta_completa, width=1400, height=800)
        
        print(f"✓ Gráfico guardado: {ruta_completa}")
        
    except Exception as e:
        print(f"⚠️  No se pudo guardar el gráfico: {e}")
        print(f"   (El gráfico se mostrará en el navegador de todas formas)")

def formatear_moneda(valor):
    """
    Formatea un número como pesos colombianos
    
    Args:
        valor (float): Valor numérico
    
    Returns:
        str: Valor formateado como moneda
    """
    return f"${valor:,.0f} COP"

def formatear_porcentaje(valor):
    """
    Formatea un número como porcentaje
    
    Args:
        valor (float): Valor numérico (0-100)
    
    Returns:
        str: Valor formateado como porcentaje
    """
    return f"{valor:.1f}%"

def formatear_numero(valor):
    """
    Formatea un número con separadores de miles
    
    Args:
        valor (float): Valor numérico
    
    Returns:
        str: Valor formateado
    """
    return f"{valor:,.0f}"

def calcular_metricas_basicas(df):
    """
    Calcula métricas básicas del dataset
    
    Args:
        df (pd.DataFrame): DataFrame con datos de Oasis
    
    Returns:
        dict: Diccionario con métricas básicas
    """
    metricas = {
        'total_transacciones': len(df),
        'total_usuarios': df['user_id'].nunique() if 'user_id' in df.columns else 0,
        'total_estaciones': df['evse_uid'].nunique() if 'evse_uid' in df.columns else 0,
        'ingresos_totales': df['amount_transaction'].sum() if 'amount_transaction' in df.columns else 0,
        'energia_total': df['energy_kwh'].sum() if 'energy_kwh' in df.columns else 0
    }
    
    return metricas

def validar_columnas(df, columnas_requeridas):
    """
    Valida que el DataFrame tenga las columnas requeridas
    
    Args:
        df (pd.DataFrame): DataFrame a validar
        columnas_requeridas (list): Lista de nombres de columnas requeridas
    
    Returns:
        bool: True si todas las columnas existen, False en caso contrario
    """
    columnas_faltantes = [col for col in columnas_requeridas if col not in df.columns]
    
    if columnas_faltantes:
        print(f"⚠️  Columnas faltantes: {', '.join(columnas_faltantes)}")
        return False
    
    return True

# Configuración de colores para gráficos
COLORES = {
    'primary': '#667eea',
    'secondary': '#764ba2',
    'success': '#48bb78',
    'warning': '#f6ad55',
    'danger': '#fc8181',
    'info': '#4299e1',
    'light': '#f7fafc',
    'dark': '#2d3748'
}

# Configuración de estilo
ESTILO_GRAFICO = {
    'font_family': 'Arial',
    'title_size': 18,
    'axis_title_size': 14,
    'bg_color': '#ffffff',
    'grid_color': '#e2e8f0'
}