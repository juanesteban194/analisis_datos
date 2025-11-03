"""
Funciones auxiliares compartidas por todos los gráficos
"""

import os
import pandas as pd

def cargar_datos(ruta='data/df_oasis_clean.csv'):
    """Carga el dataset principal de Oasis EVSE."""
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
    """Guarda un gráfico como PNG en outputs; si no hay kaleido, guarda HTML."""
    try:
        output_dir = 'outputs'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        ruta_png = os.path.join(output_dir, nombre_archivo)
        fig.write_image(ruta_png, width=1400, height=800)  # requiere kaleido
        print(f"✓ Gráfico guardado: {ruta_png}")

    except Exception as e:
        # Fallback a HTML si kaleido no está disponible
        ruta_html = os.path.join('outputs', os.path.splitext(nombre_archivo)[0] + '.html')
        fig.write_html(ruta_html)
        print(f"⚠️  No se pudo guardar PNG ({e}). Guardado HTML: {ruta_html}")

def formatear_moneda(valor):
    """Formatea un número como pesos colombianos."""
    return f"${valor:,.0f} COP"

def formatear_porcentaje(valor):
    """Formatea un número como porcentaje."""
    return f"{valor:.1f}%"

def formatear_numero(valor):
    """Formatea un número con separadores de miles."""
    return f"{valor:,.0f}"

def calcular_metricas_basicas(df):
    """Calcula métricas básicas del dataset."""
    # Asegurar tipos numéricos para evitar errores de suma si hay strings
    if 'amount_transaction' in df.columns:
        df['amount_transaction'] = pd.to_numeric(df['amount_transaction'], errors='coerce')
    if 'energy_kwh' in df.columns:
        df['energy_kwh'] = pd.to_numeric(df['energy_kwh'], errors='coerce')

    metricas = {
        'total_transacciones': len(df),
        'total_usuarios': df['user_id'].nunique() if 'user_id' in df.columns else 0,
        'total_estaciones': df['evse_uid'].nunique() if 'evse_uid' in df.columns else 0,
        'ingresos_totales': df['amount_transaction'].sum() if 'amount_transaction' in df.columns else 0,
        'energia_total': df['energy_kwh'].sum() if 'energy_kwh' in df.columns else 0,
    }
    return metricas

def validar_columnas(df, columnas_requeridas):
    """
    Valida que el DataFrame tenga las columnas requeridas.
    Devuelve True si todo está OK; si faltan columnas, imprime y devuelve False.
    """
    faltantes = [col for col in columnas_requeridas if col not in df.columns]
    if faltantes:
        print(f"⚠️  Columnas faltantes: {', '.join(faltantes)}")
        return False
    return True

# Paleta general
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

# Estilo base de gráficos
ESTILO_GRAFICO = {
    'font_family': 'Arial',
    'title_size': 18,
    'axis_title_size': 14,
    'bg_color': '#ffffff',
    'grid_color': '#e2e8f0'
}
