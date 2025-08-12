import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ----------------------------
# Configuración de la página
# ----------------------------
st.set_page_config(page_title="Simulador de Hipoteca", layout="wide")
st.title("🏠 Simulador de Hipoteca")

st.caption(
    "Introduce el importe, el plazo y el interés anual. "
    "El cálculo asume capitalización mensual (interés nominal anual / 12)."
)

# ----------------------------
# Entradas
# ----------------------------
with st.sidebar:
    st.header("Parámetros")
    principal = st.number_input(
        "Importe a financiar (Cantidad solicitada)",
        min_value=1000.0, value=150000.0, step=1000.0, format="%.2f"
    )
    years = st.slider("Plazo (años)", min_value=1, max_value=40, value=25, step=1)
    annual_rate_pct = st.number_input(
        "Interés aplicado (% TIN anual)",
        min_value=0.0, max_value=30.0, value=3.0, step=0.05, format="%.2f"
    )
    start_month = st.selectbox(
        "Mes de inicio (opcional, solo para agrupar por años)",
        options=list(range(1, 13)),
        index=0,
        format_func=lambda m: f"{m:02d}"
    )

# ----------------------------
# Cálculos base
# ----------------------------
n_months = years * 12
r_monthly = (annual_rate_pct / 100.0) / 12.0

def amortization_schedule(P: float, r: float, n: int) -> pd.DataFrame:
    """Devuelve un DataFrame con el cuadro de amortización mensual."""
    if P <= 0 or n <= 0:
        return pd.DataFrame()

    if r == 0:
        payment = P / n
    else:
        payment = P * r / (1 - (1 + r) ** (-n))

    balance = P
    rows = []

    for m in range(1, n + 1):
        if r == 0:
            interest = 0.0
            principal_pay = payment
        else:
            interest = balance * r
            principal_pay = payment - interest

        # Ajuste en la última cuota para eliminar decimales residuales
        if m == n:
            principal_pay = balance
            payment_eff = principal_pay + interest
            balance_end = 0.0
        else:
            payment_eff = payment
            balance_end = balance - principal_pay

        rows.append(
            {
                "Mes": m,
                "Cuota": payment_eff,
                "Intereses": interest,
                "Amortización": principal_pay,
                "Saldo final": max(balance_end, 0.0),
            }
        )
        balance = balance_end

    df = pd.DataFrame(rows)
    return df

df = amortization_schedule(principal, r_monthly, n_months)

if df.empty:
    st.warning("Introduce un importe y un plazo válidos.")
    st.stop()

# ----------------------------
# Métricas principales
# ----------------------------
total_interest = float(df["Intereses"].sum())
monthly_payment = float(df["Cuota"].iloc[0]) if not df.empty else 0.0

m1, m2, m3 = st.columns(3)
m1.metric("💳 Cuota mensual", f"{monthly_payment:,.2f} €")
m2.metric("💡 Intereses totales a pagar", f"{total_interest:,.2f} €")
m3.metric("🗓️ Nº de cuotas (meses)", f"{n_months}")

st.divider()

# ----------------------------
# Gráfica de tarta (Principal vs Intereses)
# ----------------------------
pie_fig = go.Figure(
    data=[
        go.Pie(
            labels=["Principal (solicitado)", "Intereses"],
            values=[principal, total_interest],
            hole=0.35,
            hovertemplate="%{label}: %{value:,.2f} €<extra></extra>",
        )
    ]
)
pie_fig.update_layout(
    title="Distribución total: principal vs intereses",
    legend=dict(orientation="h", yanchor="bottom", y=-0.05, xanchor="center", x=0.5),
)

# ----------------------------
# Gráfica de barras apiladas anual (Amortización vs Intereses)
# ----------------------------
df["Mes calendario"] = ((df["Mes"] - 1 + (start_month - 1)) % 12) + 1
df["Año"] = ((df["Mes"] - 1 + (start_month - 1)) // 12) + 1

annual = (
    df.groupby("Año", as_index=False)[["Intereses", "Amortización"]]
    .sum()
    .round(2)
)

bar_fig = go.Figure()
bar_fig.add_trace(
    go.Bar(
        x=annual["Año"],
        y=annual["Intereses"],
        name="Intereses",
        hovertemplate="Año %{x}<br>Intereses: %{y:,.2f} €<extra></extra>",
    )
)
bar_fig.add_trace(
    go.Bar(
        x=annual["Año"],
        y=annual["Amortización"],
        name="Amortización",
        hovertemplate="Año %{x}<br>Amortización: %{y:,.2f} €<extra></extra>",
    )
)
bar_fig.update_layout(
    barmode="stack",
    title="Pago anual desglosado (apilado): amortización vs intereses",
    xaxis_title="Año",
    yaxis_title="€",
)

# ----------------------------
# Layout visual
# ----------------------------
c1, c2 = st.columns([1, 1])
with c1:
    st.plotly_chart(pie_fig, use_container_width=True)
with c2:
    st.plotly_chart(bar_fig, use_container_width=True)

# ----------------------------
# Detalle (opcional): primeras 12 cuotas
# ----------------------------
with st.expander("Ver detalle de las primeras 12 cuotas"):
    st.dataframe(df.head(12).style.format({
        "Cuota": "{:,.2f} €",
        "Intereses": "{:,.2f} €",
        "Amortización": "{:,.2f} €",
        "Saldo final": "{:,.2f} €",
    }))

# ----------------------------
# Notas
# ----------------------------
st.caption(
    "Notas: Si el interés es 0%, la cuota se calcula como principal / nº de meses. "
    "Este simulador no contempla comisiones, seguros ni variaciones de tipo de interés."
)
