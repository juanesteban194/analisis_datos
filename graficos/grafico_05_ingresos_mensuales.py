# -*- coding: utf-8 -*-
"""
Gráfico 5: Ingresos Mensuales
Análisis: Evolución temporal de ingresos con tendencias y estacionalidad
Tipo: Gráfico de líneas con área y tendencia
"""

import os, sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- hacer visible utils.py si se ejecuta directo desde /graficos ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# --------------------------------------------------------------------

from utils import cargar_datos, guardar_grafico

# ===============================
# Selección de columna de montos
# ===============================
def _to_num(s):
    return pd.to_numeric(s, errors="coerce")

def seleccionar_monto(df: pd.DataFrame, debug: bool = True) -> pd.Series:
    """
    Devuelve una Serie 'monto_cop' en COP reales lista para sumar.
    Reglas de prioridad + heurística de escala (1, 10, 100, 1000, …).

    Prioridad por nombre:
      1) amount_transaction_num (as-is)
      2) amount_transaction (as-is)
      3) amount_transaction_cop (*100)   # visto en tu dataset
      4) Otras columnas candidatas + heurística

    Heurística de escala:
      - Se busca que el ticket promedio (monto/tx) quede en [1k, 300k] COP.
      - Si varias escalas encajan, se elige la más cercana a 20k (centro típico).
    """
    df = df.copy()
    # Candidatas por prioridad conocida
    if "amount_transaction_num" in df.columns:
        s = _to_num(df["amount_transaction_num"]).fillna(0)
        if debug: print("[monto] usando amount_transaction_num (x1)")
        return s

    if "amount_transaction" in df.columns:
        s = _to_num(df["amount_transaction"]).fillna(0)
        if debug: print("[monto] usando amount_transaction (x1)")
        return s

    if "amount_transaction_cop" in df.columns:
        # En tu data, esta venía 100x pequeña → *100
        s = _to_num(df["amount_transaction_cop"]).fillna(0) * 100
        if debug: print("[monto] usando amount_transaction_cop (x100)")
        return s

    # Búsqueda abierta por nombres y escalas
    keys = ["amount", "payment", "price", "total", "cost", "monto", "ingres"]
    candidates = [c for c in df.columns if any(k in c.lower() for k in keys)]

    if not candidates:
        raise ValueError("No se encontró una columna candidata de montos.")

    scales = [1000, 100, 10, 1, 0.1, 0.01, 0.001]
    best = None
    TARGET_MEAN = 20_000  # ticket promedio típico
    MIN_MEAN, MAX_MEAN = 1_000, 300_000

    for col in candidates:
        base = _to_num(df[col]).values
        base = np.where(np.isfinite(base), base, 0.0)
        # si hay demasiados ceros/NaN, descartamos
        if np.nanmax(np.abs(base)) == 0:
            continue
        for sc in scales:
            vals = base * sc
            # promedio por transacción (sobre valores > 0 para evitar sesgos de ceros)
            mask = vals > 0
            if not mask.any():
                continue
            mean_tx = vals[mask].mean()
            if MIN_MEAN <= mean_tx <= MAX_MEAN:
                score = abs(mean_tx - TARGET_MEAN)  # más cerca de 20k mejor
                if (best is None) or (score < best["score"]):
                    best = dict(col=col, scale=sc, score=score)

    if best is None:
        # fallback: usa la mejor candidata por magnitud total (x1)
        col0 = candidates[0]
        if debug: print(f"[monto] fallback usando {col0} (x1)")
        return _to_num(df[col0]).fillna(0)

    if debug:
        print(f"[monto] usando {best['col']} (x{best['scale']}) "
              f"→ ticket prom. razonable")

    return _to_num(df[best["col"]]).fillna(0) * best["scale"]


# ===============================
# Preparación de datos mensuales
# ===============================
def preparar_datos_temporales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara datos agregados por mes.
    - Asegura tipos correctos
    - Selecciona monto real (COP)
    - Agrega métricas mensuales
    """
    df = df.copy()

    # Fechas
    if "start_date_time" not in df.columns:
        raise ValueError("Falta la columna 'start_date_time'.")
    df["start_date_time"] = pd.to_datetime(df["start_date_time"], errors="coerce")
    df = df.dropna(subset=["start_date_time"])

    # Columnas opcionales que usamos si existen
    for col in ["user_id", "evse_uid", "energy_kwh"]:
        if col not in df.columns:
            df[col] = np.nan

    # Monto (COP reales) consistente con el total del dataset
    df["monto_cop"] = seleccionar_monto(df, debug=True)

    # Año-mes como Period y Timestamp (eje ordenable)
    df["year_month"] = df["start_date_time"].dt.to_period("M")
    df["year_month_dt"] = df["year_month"].dt.to_timestamp()  # 1er día del mes

    # Agregación mensual
    monthly = (
        df.groupby("year_month")
          .agg(
              year_month_dt=("year_month_dt", "first"),
              ingresos_totales=("monto_cop", "sum"),
              ingreso_promedio=("monto_cop", "mean"),
              num_transacciones=("monto_cop", "count"),
              usuarios_activos=("user_id", pd.Series.nunique),
              estaciones_activas=("evse_uid", pd.Series.nunique),
              energia_total_kwh=("energy_kwh", "sum"),
          )
          .reset_index(drop=False)
          .sort_values("year_month")
    )

    # Métricas derivadas (evitar división por cero)
    monthly["ingreso_por_usuario"] = np.where(
        monthly["usuarios_activos"].fillna(0) > 0,
        monthly["ingresos_totales"] / monthly["usuarios_activos"].replace(0, np.nan),
        np.nan,
    )

    monthly["ingreso_por_transaccion"] = np.where(
        monthly["num_transacciones"].fillna(0) > 0,
        monthly["ingresos_totales"] / monthly["num_transacciones"].replace(0, np.nan),
        np.nan,
    )

    # Crecimientos %
    monthly["crecimiento_ingresos_pct"] = monthly["ingresos_totales"].pct_change() * 100
    monthly["crecimiento_usuarios_pct"] = monthly["usuarios_activos"].pct_change() * 100

    # Label
    monthly["year_month_str"] = monthly["year_month"].astype(str)

    return monthly


def calcular_tendencia(data: pd.DataFrame, columna: str) -> np.ndarray:
    """Regresión lineal simple sobre la serie seleccionada."""
    y = data[columna].astype(float).fillna(0).values
    x = np.arange(len(data))
    if len(data) >= 2 and not np.allclose(y, y[0]):
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        return p(x)
    return np.repeat(np.nanmean(y), len(y))


# ===============================
# Gráficos
# ===============================
def crear_grafico(monthly_data: pd.DataFrame) -> go.Figure:
    tendencia_ingresos = calcular_tendencia(monthly_data, "ingresos_totales")

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            "Evolución de Ingresos Mensuales",
            "Usuarios Activos y Transacciones por Mes",
            "Crecimiento Mensual (%)"
        ),
        vertical_spacing=0.1,
        row_heights=[0.4, 0.3, 0.3]
    )

    # 1) Ingresos (área) + tendencia
    fig.add_trace(
        go.Scatter(
            x=monthly_data["year_month_dt"],
            y=monthly_data["ingresos_totales"],
            mode="lines+markers",
            name="Ingresos",
            line=dict(color="#3b82f6", width=3),
            marker=dict(size=8, color="#3b82f6"),
            fill="tozeroy",
            fillcolor="rgba(59,130,246,0.2)",
            hovertemplate="<b>%{x|%Y-%m}</b><br>Ingresos: $%{y:,.0f} COP<extra></extra>",
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=monthly_data["year_month_dt"],
            y=tendencia_ingresos,
            mode="lines",
            name="Tendencia",
            line=dict(color="#ef4444", width=2, dash="dash"),
            hovertemplate="<b>%{x|%Y-%m}</b><br>Tendencia: $%{y:,.0f} COP<extra></extra>",
        ),
        row=1, col=1
    )

    # 2) Usuarios y transacciones
    fig.add_trace(
        go.Scatter(
            x=monthly_data["year_month_dt"],
            y=monthly_data["usuarios_activos"],
            mode="lines+markers",
            name="Usuarios Activos",
            line=dict(color="#10b981", width=2),
            marker=dict(size=6),
            hovertemplate="<b>%{x|%Y-%m}</b><br>Usuarios: %{y}<extra></extra>",
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=monthly_data["year_month_dt"],
            y=monthly_data["num_transacciones"],
            mode="lines+markers",
            name="Transacciones",
            line=dict(color="#8b5cf6", width=2),
            marker=dict(size=6),
            hovertemplate="<b>%{x|%Y-%m}</b><br>Transacciones: %{y}<extra></extra>",
        ),
        row=2, col=1
    )

    # 3) Crecimiento de ingresos (%)
    colors = ["#10b981" if v >= 0 else "#ef4444"
              for v in monthly_data["crecimiento_ingresos_pct"].fillna(0)]
    fig.add_trace(
        go.Bar(
            x=monthly_data["year_month_dt"],
            y=monthly_data["crecimiento_ingresos_pct"].fillna(0),
            name="Crecimiento Ingresos",
            marker_color=colors,
            hovertemplate="<b>%{x|%Y-%m}</b><br>Crecimiento: %{y:.1f}%<extra></extra>",
        ),
        row=3, col=1
    )

    # Layout
    fig.update_layout(
        title=dict(
            text="Análisis Temporal de Ingresos - Evolución Mensual",
            x=0.5, xanchor="center",
            font=dict(size=20, color="#2d3748", family="Arial Black")
        ),
        showlegend=True,
        height=1000,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=12),
        hovermode="x unified",
        margin=dict(t=80, r=40, b=40, l=60),
    )

    # Ejes
    for r in (1, 2, 3):
        fig.update_xaxes(
            tickangle=-45, gridcolor="#e2e8f0",
            tickformat="%Y-%m",
            row=r, col=1
        )

    fig.update_yaxes(title_text="Ingresos (COP)", gridcolor="#e2e8f0", row=1, col=1)
    fig.update_yaxes(title_text="Usuarios / Transacciones", gridcolor="#e2e8f0", row=2, col=1)
    fig.update_yaxes(title_text="Crecimiento (%)", gridcolor="#e2e8f0",
                     zeroline=True, zerolinecolor="#94a3b8", row=3, col=1)

    return fig


def crear_grafico_comparativo(monthly_data: pd.DataFrame) -> go.Figure:
    """Comparación de métricas normalizadas (0-100). Maneja series constantes."""
    def normalize(s: pd.Series) -> pd.Series:
        s = s.astype(float).fillna(0)
        rng = s.max() - s.min()
        if rng == 0:
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - s.min()) / rng * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly_data["year_month_dt"],
        y=normalize(monthly_data["ingresos_totales"]),
        mode="lines+markers", name="Ingresos", line=dict(color="#3b82f6", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=monthly_data["year_month_dt"],
        y=normalize(monthly_data["usuarios_activos"]),
        mode="lines+markers", name="Usuarios", line=dict(color="#10b981", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=monthly_data["year_month_dt"],
        y=normalize(monthly_data["num_transacciones"]),
        mode="lines+markers", name="Transacciones", line=dict(color="#8b5cf6", width=2)
    ))

    fig.update_layout(
        title=dict(
            text="Comparación de Métricas Normalizadas (0–100)",
            x=0.5, xanchor="center",
            font=dict(size=18, color="#2d3748", family="Arial Black")
        ),
        xaxis_title="Mes",
        yaxis_title="Índice Normalizado (0–100)",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=500,
        font=dict(family="Arial", size=12),
        hovermode="x unified"
    )
    fig.update_xaxes(tickangle=-45, gridcolor="#e2e8f0", tickformat="%Y-%m")
    fig.update_yaxes(gridcolor="#e2e8f0")
    return fig


# ===============================
# Insights
# ===============================
def analizar(monthly_data: pd.DataFrame) -> dict:
    total_ingresos = float(monthly_data["ingresos_totales"].sum())
    total_transacciones = int(monthly_data["num_transacciones"].sum())

    ingreso_promedio_mes = float(monthly_data["ingresos_totales"].mean())
    usuarios_promedio_mes = float(monthly_data["usuarios_activos"].mean())

    mejor_mes = monthly_data.loc[monthly_data["ingresos_totales"].idxmax()]
    peor_mes  = monthly_data.loc[monthly_data["ingresos_totales"].idxmin()]

    primer_mes = float(monthly_data.iloc[0]["ingresos_totales"])
    ultimo_mes = float(monthly_data.iloc[-1]["ingresos_totales"])
    crecimiento_total = ( (ultimo_mes - primer_mes) / primer_mes * 100 ) if primer_mes != 0 else 0.0

    # Tendencia (pendiente)
    y = monthly_data["ingresos_totales"].astype(float).fillna(0).values
    x = np.arange(len(monthly_data))
    z = np.polyfit(x, y, 1) if len(monthly_data) >= 2 else (0, 0)
    tendencia_pendiente = z[0] if isinstance(z, np.ndarray) else 0.0

    volatilidad = (monthly_data["ingresos_totales"].std() / ingreso_promedio_mes * 100) if ingreso_promedio_mes != 0 else 0.0

    meses_positivos = int((monthly_data["crecimiento_ingresos_pct"] > 0).sum())
    total_meses = max(0, len(monthly_data) - 1)

    insights = {
        "total_meses": len(monthly_data),
        "total_ingresos": int(round(total_ingresos)),
        "total_transacciones": total_transacciones,

        "ingreso_promedio_mes": int(round(ingreso_promedio_mes)),
        "usuarios_promedio_mes": int(round(usuarios_promedio_mes)),

        "mejor_mes_fecha": mejor_mes["year_month"].strftime("%Y-%m"),
        "mejor_mes_ingresos": int(round(mejor_mes["ingresos_totales"])),

        "peor_mes_fecha": peor_mes["year_month"].strftime("%Y-%m"),
        "peor_mes_ingresos": int(round(peor_mes["ingresos_totales"])),

        "crecimiento_total": float(crecimiento_total),
        "tendencia": "positiva" if tendencia_pendiente > 0 else "negativa",
        "tendencia_valor": float(tendencia_pendiente),

        "volatilidad": float(volatilidad),

        "meses_crecimiento_positivo": meses_positivos,
        "pct_meses_positivos": (meses_positivos / total_meses * 100) if total_meses > 0 else 0.0,

        "insight_tendencia": f"La tendencia general es {'positiva' if tendencia_pendiente > 0 else 'negativa'} "
                             f"con un crecimiento {'sostenido' if crecimiento_total > 0 else 'negativo'} "
                             f"de {crecimiento_total:.1f}% desde el inicio.",

        "insight_mejor_peor": (
            f"El mejor mes fue {mejor_mes['year_month'].strftime('%Y-%m')} "
            f"(${mejor_mes['ingresos_totales']:,.0f} COP) y el peor fue "
            f"{peor_mes['year_month'].strftime('%Y-%m')} "
            f"(${peor_mes['ingresos_totales']:,.0f} COP), una diferencia de "
            f"{((mejor_mes['ingresos_totales'] - peor_mes['ingresos_totales']) / max(1, peor_mes['ingresos_totales']) * 100):.1f}%."
        ),

        "insight_volatilidad": (
            f"La volatilidad de ingresos es de {volatilidad:.1f}%, "
            f"indicando {'alta' if volatilidad > 30 else 'moderada' if volatilidad > 15 else 'baja'} variabilidad mensual."
        ),

        "insight_consistencia": (
            f"{meses_positivos} de {total_meses} meses "
            f"({(meses_positivos/total_meses*100):.1f}%) mostraron crecimiento positivo, "
            f"{'demostrando consistencia' if (total_meses>0 and meses_positivos/total_meses > 0.5) else 'indicando inconsistencia'}."
        ),
    }
    return insights


# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("GRÁFICO 5: INGRESOS MENSUALES")
    print("="*80)

    # Cargar datos (usa el dataset listo)
    csv_path = os.path.join(PROJECT_ROOT, "data", "df_oasis_ready.csv")
    df = cargar_datos(csv_path)
    print(f"✓ Datos cargados: {len(df):,} registros\n")

    # Preparar/agrupar
    print("Preparando datos mensuales...")
    monthly_data = preparar_datos_temporales(df)
    total_check = int(monthly_data["ingresos_totales"].sum())
    print(f"✓ Total verificado por suma mensual: ${total_check:,} COP\n")

    # Gráficos
    print("Creando gráfico principal...")
    fig_main = crear_grafico(monthly_data)

    print("Creando gráfico comparativo...")
    fig_comp = crear_grafico_comparativo(monthly_data)

    # Insights
    print("\n" + "="*80)
    print("ANÁLISIS TEMPORAL")
    print("="*80)
    insights = analizar(monthly_data)

    print(f"\nRESUMEN GENERAL:")
    print(f"  • Período analizado: {insights['total_meses']} meses")
    print(f"  • Ingresos totales: ${insights['total_ingresos']:,} COP")
    print(f"  • Transacciones totales: {insights['total_transacciones']:,}")
    print(f"  • Ingreso promedio/mes: ${insights['ingreso_promedio_mes']:,} COP")
    print(f"  • Usuarios promedio/mes: {insights['usuarios_promedio_mes']}")

    print(f"\nMEJOR Y PEOR MES:")
    print(f"  • Mejor: {insights['mejor_mes_fecha']} - ${insights['mejor_mes_ingresos']:,} COP")
    print(f"  • Peor:  {insights['peor_mes_fecha']} - ${insights['peor_mes_ingresos']:,} COP")

    print(f"\nTENDENCIA:")
    print(f"  • Crecimiento total: {insights['crecimiento_total']:.1f}%")
    print(f"  • Tendencia: {insights['tendencia'].upper()}")
    print(f"  • Volatilidad: {insights['volatilidad']:.1f}%")

    print(f"\nCONSISTENCIA:")
    total_meses_ex_crec = max(0, insights['total_meses'] - 1)
    print(f"  • Meses con crecimiento: {insights['meses_crecimiento_positivo']} de {total_meses_ex_crec}")
    print(f"  • Porcentaje positivo: {insights['pct_meses_positivos']:.1f}%")

    print(f"\nINSIGHTS PRINCIPALES:")
    print(f"  1) {insights['insight_tendencia']}")
    print(f"  2) {insights['insight_mejor_peor']}")
    print(f"  3) {insights['insight_volatilidad']}")
    print(f"  4) {insights['insight_consistencia']}")

    # Guardar y mostrar
    print("\n" + "="*80)
    print("Guardando gráficos...")
    guardar_grafico(fig_main, "grafico_05_ingresos_mensuales.png")
    guardar_grafico(fig_comp, "grafico_05_comparativo.png")

    print("Abriendo gráfico principal...")
    fig_main.show()
    print("Abriendo gráfico comparativo...")
    fig_comp.show()
