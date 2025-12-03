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
# ESTILOS (Tabs + Parámetros sticky + caja gris + valores grandes)
# ----------------------------
st.markdown(
    """
    <style>
    /* ===== Tabs mejoradas ===== */
    .stTabs [role="tablist"] {
        gap: 20px;
        justify-content: center;
        border-bottom: 3px solid #4A90E2;
        margin-bottom: 1rem;
    }
    .stTabs [role="tab"] {
        background-color: #f5f7fa;
        padding: 0.8rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 12px 12px 0 0;
        border: 2px solid #d0d0d0;
        border-bottom: none;
        color: #333;
        transition: all 0.3s ease;
    }
    .stTabs [role="tab"]:hover {
        background-color: #e8f0fe;
        border-color: #4A90E2;
        color: #000;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4A90E2 !important;
        color: white !important;
        border-color: #4A90E2 !important;
        font-weight: 700 !important;
    }

    /* ===== Card sticky para parámetros ===== */
    div[data-testid="stForm"] {
        position: sticky;
        top: 0;
        z-index: 1000;
        background: #ffffff;
        border: 2px solid #e6e9ef;
        border-radius: 16px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.06);
        padding: 1rem 1.25rem;
        margin-bottom: 1.25rem;
    }
    .param-header {
        display: flex;
        align-items: center;
        gap: .6rem;
        margin-bottom: .5rem;
    }
    .param-chip {
        background: #4A90E2;
        color: #fff;
        font-weight: 700;
        font-size: .85rem;
        padding: .25rem .6rem;
        border-radius: 999px;
    }
    .param-subtle {
        color: #5f6570;
        font-size: .9rem;
        margin-left: .25rem;
    }
    .stButton>button {
        border-radius: 10px;
        padding: .6rem 1rem;
        font-weight: 700;
    }

    /* Caja gris suave para resaltar el interés del periodo 2 */
    .soft-box {
        background: #f2f3f5;
        border: 1px solid #e1e3e8;
        border-radius: 12px;
        padding: 0.8rem 1rem;
        display: block;
    }

    /* Títulos y valores grandes (similar a st.metric) */
    .value-title {
        font-size: 0.95rem;
        color: #5f6570;
        margin-bottom: .25rem;
    }
    .value-big {
        font-size: 1.6rem;
        font-weight: 800;
        line-height: 1.1;
    }

    /* ===== Fix responsive tabs (móvil) ===== */
    @media (max-width: 640px) {
      .stTabs [role="tablist"] {
        justify-content: flex-start;
        gap: 8px;
        overflow-x: auto;
        overflow-y: hidden;
        padding-bottom: .25rem;
        scrollbar-width: thin;
      }
      .stTabs [role="tab"] {
        flex: 0 0 auto;
        white-space: nowrap;
        padding: .5rem .9rem;
        font-size: .95rem;
        border-radius: 10px 10px 0 0;
      }
      div[data-testid="stForm"] {
        top: .5rem;
        padding: .75rem .9rem;
        margin-bottom: .75rem;
      }
    }
    .stTabs [role="tablist"]::-webkit-scrollbar { height: 6px; }
    .stTabs [role="tablist"]::-webkit-scrollbar-thumb {
      border-radius: 999px;
      background: #cdd6e1;
    }

    /* ===== Uniformar métricas en móvil ===== */
    @media (max-width: 640px) {
      .soft-box {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
      }
      .value-title {
        margin-top: .35rem;
      }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================
# Utilidades
# ============================
def eur(x: float) -> str:
    s = f"{x:,.2f} €"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")

def amortization_schedule(P: float, r_m: float, n: int) -> pd.DataFrame:
    """Cuadro de amortización con tipo mensual constante r_m durante n meses."""
    if P <= 0 or n <= 0:
        return pd.DataFrame()
    if r_m == 0:
        payment = P / n
    else:
        payment = P * r_m / (1 - (1 + r_m) ** (-n))

    balance = P
    rows = []
    for m in range(1, n + 1):
        interest = balance * r_m if r_m != 0 else 0.0
        principal_pay = payment - interest if r_m != 0 else payment

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

    return pd.DataFrame(rows)

def mixed_total_interest(P: float, n: int, r1_m: float, m1: int, r2_m: float):
    """
    Intereses totales de una hipoteca mixta:
    - Periodo 1: r1_m, cuota calculada con r1_m para TODO el plazo (n), se pagan m1 meses.
    - Periodo 2: r2_m, cuota recalculada con r2_m sobre saldo restante y n-m1 meses.
    Devuelve (intereses_totales, intereses_periodo1, intereses_periodo2, saldo_tras_p1).
    """
    if P <= 0 or n <= 0 or m1 < 0 or m1 > n:
        return 0.0, 0.0, 0.0, P

    # Periodo 1
    if r1_m == 0:
        payment1 = P / n
    else:
        payment1 = P * r1_m / (1 - (1 + r1_m) ** (-n))

    balance = P
    interest_p1 = 0.0
    for _ in range(m1):
        interest = balance * r1_m if r1_m != 0 else 0.0
        principal_pay = payment1 - interest if r1_m != 0 else payment1
        balance -= principal_pay
        interest_p1 += interest

    n2 = n - m1
    if n2 <= 0:
        return interest_p1, interest_p1, 0.0, 0.0

    # Periodo 2
    if r2_m == 0:
        payment2 = balance / n2
    else:
        payment2 = balance * r2_m / (1 - (1 + r2_m) ** (-n2))

    interest_p2 = 0.0
    bal = balance
    for _ in range(n2):
        interest = bal * r2_m if r2_m != 0 else 0.0
        principal_pay = payment2 - interest if r2_m != 0 else payment2
        bal -= principal_pay
        interest_p2 += interest

    return interest_p1 + interest_p2, interest_p1, interest_p2, balance

def solve_r2_for_equal_interest(P: float, n: int, r_fixed_m: float, r1_m: float, m1: int):
    """
    Encuentra r2_m (tipo mensual periodo 2) tal que:
    intereses_totales_mixta(r1_m, m1, r2_m) == intereses_totales_fija(r_fixed_m)
    Búsqueda por bisección con cota amplia.
    """
    df_fixed = amortization_schedule(P, r_fixed_m, n)
    target = float(df_fixed["Intereses"].sum())

    if m1 >= n:
        total_mixed, ip1, ip2, _ = mixed_total_interest(P, n, r1_m, m1, r2_m=0.0)
        return None, target, total_mixed, ip1, ip2

    def f(r2m):
        total, _, _, _ = mixed_total_interest(P, n, r1_m, m1, r2m)
        return total - target

    lo = 0.0
    hi = 2.0 / 12.0  # 200% anual aprox.
    f_lo = f(lo)
    f_hi = f(hi)

    attempts = 0
    while f_lo * f_hi > 0 and attempts < 20:
        hi *= 1.5
        f_hi = f(hi)
        attempts += 1

    if f_lo * f_hi > 0:
        total_mixed, ip1, ip2, _ = mixed_total_interest(P, n, r1_m, m1, r2_m=lo)
        return None, target, total_mixed, ip1, ip2

    for _ in range(80):
        mid = (lo + hi) / 2
        f_mid = f(mid)
        if abs(f_mid) < 1e-8:
            lo = hi = mid
            break
        if f_lo * f_mid <= 0:
            hi = mid
        else:
            lo = mid
            f_lo = f_mid

    r2_m_solution = (lo + hi) / 2
    total_mixed, ip1, ip2, _ = mixed_total_interest(P, n, r1_m, m1, r2_m_solution)
    return r2_m_solution, target, total_mixed, ip1, ip2

# ============================
# Matriz prima orientativa (ING) + interpolación (edad x capital)
# ============================
CAPITALS_ING = [50000, 75000, 100000, 125000, 150000, 175000, 200000, 225000, 250000, 275000, 300000, 325000, 350000, 375000, 400000]

PREMIAS_ING = {
    18: [9.41, 13.81, 18.64, 22.88, 27.43, 31.96, 36.50, 41.25, 45.83, 50.42, 54.56, 59.19, 63.74, 68.27, 72.81],
    19: [9.25, 13.88, 18.60, 23.13, 27.75, 32.38, 37.00, 41.63, 46.26, 50.88, 55.39, 60.13, 64.76, 69.39, 73.79],
    20: [9.16, 13.73, 18.31, 22.89, 27.69, 32.05, 36.70, 41.20, 45.78, 50.36, 54.74, 59.51, 64.09, 68.67, 72.77],
    21: [9.20, 13.78, 18.38, 22.98, 27.75, 32.17, 36.83, 41.36, 45.96, 50.56, 54.99, 59.74, 64.34, 68.94, 73.15],
    22: [9.23, 13.84, 18.45, 23.07, 27.82, 32.30, 36.96, 41.52, 46.14, 50.75, 55.25, 59.98, 64.58, 69.21, 73.53],
    23: [9.27, 13.89, 18.52, 23.16, 27.88, 32.42, 37.08, 41.69, 46.32, 50.95, 55.50, 60.21, 64.83, 69.47, 73.91],
    24: [9.30, 13.95, 18.60, 23.25, 27.95, 32.55, 37.21, 41.85, 46.49, 51.14, 55.76, 60.45, 65.09, 69.74, 74.29],
    25: [9.34, 14.00, 18.67, 23.34, 28.01, 32.67, 37.34, 42.01, 46.67, 51.34, 56.01, 60.68, 65.35, 70.01, 74.68],
    26: [9.27, 13.90, 18.60, 23.18, 27.83, 32.46, 37.04, 41.68, 46.34, 51.00, 55.47, 60.04, 64.75, 69.38, 74.11],
    27: [9.21, 13.81, 18.54, 23.01, 27.64, 32.25, 36.74, 41.36, 46.00, 50.67, 54.94, 59.41, 64.15, 68.74, 73.53],
    28: [9.14, 13.71, 18.47, 22.85, 27.46, 32.04, 36.44, 41.04, 45.67, 50.34, 54.40, 58.88, 63.55, 68.11, 72.96],
    29: [9.08, 13.62, 18.41, 22.69, 27.27, 31.82, 36.13, 40.72, 45.33, 50.00, 53.87, 58.36, 62.95, 67.48, 71.89],
    30: [9.01, 13.52, 18.34, 22.53, 27.09, 31.60, 35.83, 40.36, 45.00, 49.50, 53.33, 57.84, 62.34, 66.85, 70.82],
    31: [9.29, 13.95, 19.03, 23.24, 27.81, 32.59, 37.04, 41.71, 46.43, 50.93, 55.11, 59.77, 64.42, 69.08, 73.18],
    32: [9.58, 14.39, 19.71, 23.94, 28.54, 33.57, 38.26, 43.07, 47.86, 52.37, 56.89, 61.71, 66.50, 71.32, 75.55],
    33: [9.86, 14.82, 20.40, 24.64, 29.28, 34.56, 39.48, 44.43, 49.29, 53.80, 58.68, 63.64, 68.59, 73.56, 77.91],
    34: [10.15, 15.26, 21.09, 25.34, 30.01, 35.54, 40.69, 45.77, 50.72, 55.23, 60.46, 65.57, 70.67, 75.79, 80.26],
    35: [10.43, 15.65, 21.80, 26.09, 31.69, 36.52, 41.87, 47.11, 52.17, 57.42, 62.25, 67.50, 72.76, 78.01, 82.62],
    36: [11.01, 16.51, 23.01, 27.52, 33.55, 38.81, 44.32, 49.84, 55.03, 60.56, 65.87, 71.38, 76.81, 82.34, 87.42],
    37: [11.58, 17.37, 24.22, 28.95, 35.40, 41.09, 46.76, 52.56, 57.90, 63.71, 69.48, 75.27, 80.86, 86.66, 92.23],
    38: [12.16, 18.23, 25.44, 30.38, 37.26, 43.38, 49.21, 55.29, 60.77, 66.55, 73.10, 79.15, 84.90, 90.97, 97.04],
    39: [12.73, 19.09, 26.65, 31.81, 39.12, 45.67, 51.65, 58.01, 63.64, 69.84, 76.73, 83.02, 88.94, 95.32, 101.83],
    40: [13.30, 19.95, 27.85, 33.24, 40.98, 47.96, 54.10, 60.74, 66.48, 73.13, 80.36, 86.89, 93.03, 99.66, 106.61],
    41: [15.04, 22.57, 31.45, 37.60, 46.16, 54.00, 60.91, 68.41, 74.40, 82.72, 90.44, 97.14, 104.46, 112.76, 120.96],
    42: [16.79, 25.20, 35.05, 41.95, 51.34, 60.05, 67.72, 76.09, 82.32, 92.31, 100.51, 107.38, 115.89, 125.86, 135.31],
    43: [18.53, 27.83, 38.66, 46.32, 56.51, 66.09, 74.54, 83.76, 90.24, 101.90, 110.59, 117.63, 127.33, 138.96, 149.65],
    44: [20.27, 30.45, 42.27, 50.68, 61.69, 72.14, 81.35, 91.43, 98.16, 111.51, 120.68, 127.88, 138.76, 152.06, 163.00],
    45: [22.02, 33.03, 45.88, 55.04, 66.87, 78.18, 88.17, 99.09, 110.10, 121.11, 130.76, 143.13, 154.14, 165.16, 173.35],
    46: [23.70, 35.55, 49.42, 59.25, 71.99, 83.28, 94.93, 106.68, 118.52, 130.26, 140.74, 153.71, 165.55, 177.40, 186.56],
    47: [25.37, 38.07, 52.95, 63.46, 77.12, 88.38, 101.68, 114.27, 126.95, 139.42, 150.73, 164.30, 176.97, 189.63, 199.77],
    48: [27.04, 40.60, 56.49, 67.67, 82.25, 93.47, 108.44, 121.86, 135.63, 148.58, 160.71, 174.89, 188.39, 201.87, 212.99],
    49: [28.72, 43.12, 60.03, 71.88, 87.38, 98.57, 115.21, 129.45, 144.31, 157.85, 170.70, 185.47, 199.78, 214.10, 226.20],
    50: [30.44, 45.65, 63.57, 76.09, 93.07, 108.67, 121.98, 137.04, 152.19, 167.13, 180.69, 196.03, 211.18, 226.34, 239.41],
    51: [35.69, 53.52, 73.56, 89.21, 107.55, 126.61, 141.14, 160.64, 178.43, 196.25, 209.98, 230.64, 248.33, 266.13, 272.85],
    52: [40.94, 61.39, 83.55, 102.33, 122.04, 144.54, 160.30, 184.24, 204.66, 225.36, 239.26, 265.26, 285.48, 305.92, 306.28],
    53: [46.18, 69.27, 93.54, 115.44, 136.52, 162.48, 179.46, 207.83, 230.90, 254.48, 268.55, 299.88, 322.64, 345.71, 339.70],
    54: [51.43, 77.14, 103.53, 128.56, 151.01, 180.42, 198.62, 231.43, 257.13, 283.09, 297.84, 334.50, 359.79, 385.49, 373.15],
    55: [56.67, 85.01, 113.53, 141.68, 165.45, 198.35, 217.78, 255.03, 283.37, 311.70, 322.13, 369.07, 396.93, 425.27, 426.60],
    56: [59.65, 89.47, 120.36, 149.12, 175.45, 208.76, 231.85, 268.81, 298.24, 328.06, 341.41, 387.87, 417.71, 447.53, 455.75],
    57: [62.62, 93.94, 127.18, 156.56, 185.45, 219.17, 245.92, 282.60, 313.12, 344.41, 360.68, 406.67, 438.50, 469.79, 484.91],
    58: [65.60, 98.40, 134.01, 164.00, 195.44, 229.59, 259.99, 296.38, 327.99, 360.75, 379.95, 425.47, 459.29, 492.05, 514.06],
    59: [68.57, 102.86, 140.83, 171.43, 205.44, 240.00, 272.56, 310.17, 342.86, 377.10, 399.22, 444.27, 480.07, 514.32, 543.21],
    60: [71.55, 107.32, 147.65, 178.86, 215.44, 250.41, 283.14, 321.95, 357.73, 393.50, 418.54, 465.04, 500.82, 536.59, 572.36],
}

PRIMA_ING_DF = pd.DataFrame.from_dict(PREMIAS_ING, orient="index", columns=CAPITALS_ING).sort_index()
PRIMA_ING_DF.index.name = "Edad"

def _lerp(x0, x1, y0, y1, x):
    if x0 == x1:
        return float(y0)
    return float(y0 + (y1 - y0) * (x - x0) / (x1 - x0))

def prima_orientativa_ing(edad: float, capital: float, df: pd.DataFrame) -> float:
    """Interpolación bilineal (edad x capital). Extrapola por el último tramo si se sale del rango."""
    ages = df.index.to_numpy(dtype=float)
    caps = np.array(df.columns, dtype=float)

    if edad <= ages.min():
        a0, a1 = ages[0], ages[1]
    elif edad >= ages.max():
        a0, a1 = ages[-2], ages[-1]
    else:
        a1 = ages[ages >= edad].min()
        a0 = ages[ages <= edad].max()

    if capital <= caps.min():
        c0, c1 = caps[0], caps[1]
    elif capital >= caps.max():
        c0, c1 = caps[-2], caps[-1]
    else:
        c1 = caps[caps >= capital].min()
        c0 = caps[caps <= capital].max()

    v_a0c0 = float(df.loc[int(a0), int(c0)])
    v_a0c1 = float(df.loc[int(a0), int(c1)])
    v_a1c0 = float(df.loc[int(a1), int(c0)])
    v_a1c1 = float(df.loc[int(a1), int(c1)])

    v0 = _lerp(c0, c1, v_a0c0, v_a0c1, capital)
    v1 = _lerp(c0, c1, v_a1c0, v_a1c1, capital)
    v = _lerp(a0, a1, v0, v1, edad)
    return float(v)

# ----------------------------
# PESTAÑAS
# ----------------------------
tab_simulador, tab_bonif, tab_comparador, tab_publicidad, tab_inversion = st.tabs(
    ["📊 Simulador", "🎁 Estudio Bonificaciones", "📐 Comparador: Fija vs Mixta", "🖼️ Publicidad", "💹 Analiza Inversión"]
)

# =========
# TAB 1: Simulador
# =========
with tab_simulador:
    st.markdown(
        """
        <div class="param-header">
          <span class="param-chip">Parámetros</span>
          <span class="param-subtle">Configura el simulador y pulsa “Aplicar parámetros”.</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.form("params_form_sim", clear_on_submit=False):
        c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])
        with c1:
            principal = st.number_input(
                "Importe a financiar (Cantidad solicitada)",
                min_value=1000.0, value=150000.0, step=1000.0, format="%.2f", key="p_sim"
            )
        with c2:
            years = st.slider("Plazo (años)", min_value=1, max_value=40, value=25, step=1, key="y_sim")
        with c3:
            annual_rate_pct = st.number_input(
                "Interés aplicado (% TIN anual)",
                min_value=0.0, max_value=30.0, value=3.0, step=0.05, format="%.2f", key="r_sim"
            )
        with c4:
            start_month = st.selectbox(
                "Mes de inicio (agrupación anual)",
                options=list(range(1, 13)), index=0, format_func=lambda m: f"{m:02d}", key="m_sim"
            )
        _ = st.form_submit_button("✅ Aplicar parámetros")

    n_months = years * 12
    r_monthly = (annual_rate_pct / 100.0) / 12.0

    df = amortization_schedule(principal, r_monthly, n_months)
    if df.empty:
        st.warning("Introduce un importe y un plazo válidos.")
        st.stop()

    total_interest = float(df["Intereses"].sum())
    monthly_payment = float(df["Cuota"].iloc[0])

    m1c, m2c, m3c = st.columns(3)
    m1c.metric("💳 Cuota mensual", eur(monthly_payment))
    m2c.metric("💡 Intereses totales a pagar", eur(total_interest))
    m3c.metric("🗓️ Nº de cuotas (meses)", f"{n_months}")

    st.divider()

    pie_fig = go.Figure(data=[go.Pie(labels=["Principal", "Intereses"], values=[principal, total_interest], hole=0.35)])
    pie_fig.update_layout(
        title="Distribución total: principal vs intereses",
        legend=dict(orientation="h", yanchor="bottom", y=-0.05, xanchor="center", x=0.5),
    )

    df["Mes calendario"] = ((df["Mes"] - 1 + (start_month - 1)) % 12) + 1
    df["Año"] = ((df["Mes"] - 1 + (start_month - 1)) // 12) + 1
    annual = df.groupby("Año", as_index=False)[["Intereses", "Amortización"]].sum().round(2)

    bar_fig = go.Figure()
    bar_fig.add_trace(go.Bar(x=annual["Año"], y=annual["Intereses"], name="Intereses"))
    bar_fig.add_trace(go.Bar(x=annual["Año"], y=annual["Amortización"], name="Amortización"))
    bar_fig.update_layout(
        barmode="stack",
        title="Pago anual desglosado (apilado): amortización vs intereses",
        xaxis_title="Año",
        yaxis_title="€",
    )

    c1g, c2g = st.columns([1, 1])
    with c1g:
        st.plotly_chart(pie_fig, use_container_width=True)
    with c2g:
        st.plotly_chart(bar_fig, use_container_width=True)

    with st.expander("Ver detalle de las primeras 12 cuotas"):
        st.dataframe(
            df.head(12).style.format(
                {"Cuota": eur, "Intereses": eur, "Amortización": eur, "Saldo final": eur}
            )
        )

    st.caption("Notas: Este simulador no contempla comisiones, seguros ni variaciones de tipo de interés.")

# =========
# TAB 2: Estudio Bonificaciones
# =========
with tab_bonif:
    st.markdown(
        """
        <div class="param-header">
          <span class="param-chip">Mi hipoteca</span>
          <span class="param-subtle">
            Estos parámetros son <strong>los de mi hipoteca</strong> (sin bonificaciones) y sirven como referencia.
          </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.form("params_form_bonif", clear_on_submit=False):
        c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])
        with c1:
            principal_b = st.number_input(
                "Importe a financiar (Cantidad solicitada)",
                min_value=1000.0, value=150000.0, step=1000.0, format="%.2f", key="p_bon"
            )
        with c2:
            years_b = st.slider("Plazo (años)", min_value=1, max_value=40, value=25, step=1, key="y_bon")
        with c3:
            annual_rate_pct_b = st.number_input(
                "Interés aplicado (% TIN anual)",
                min_value=0.0, max_value=30.0, value=3.0, step=0.05, format="%.2f", key="r_bon"
            )
        with c4:
            _ = st.selectbox(
                "Mes de inicio (agrupación anual)",
                options=list(range(1, 13)), index=0, format_func=lambda m: f"{m:02d}", key="m_bon"
            )
        _ = st.form_submit_button("✅ Aplicar parámetros")

    n_months_b = years_b * 12
    r_monthly_b = (annual_rate_pct_b / 100.0) / 12.0
    df_base = amortization_schedule(principal_b, r_monthly_b, n_months_b)

    if df_base.empty:
        st.warning("Introduce un importe y un plazo válidos.")
        st.stop()

    monthly_payment_base = float(df_base["Cuota"].iloc[0])

    m1c, m2c, m3c = st.columns(3)
    m1c.metric("💳 Cuota mensual (sin bonificar)", eur(monthly_payment_base))
    m2c.metric("📌 TIN anual (sin bonificar)", f"{annual_rate_pct_b:.2f} %")
    m3c.metric("🗓️ Nº de cuotas (meses)", f"{n_months_b}")

    st.divider()

    st.markdown(
        """
        <div class="param-header">
          <span class="param-chip">Bonificaciones</span>
          <span class="param-subtle">
            Introduce bonificaciones en <strong>puntos porcentuales</strong> sobre el TIN.
            Ejemplo: <strong>0,15</strong> significa que el TIN baja <strong>0,15%</strong>.
          </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.form("params_form_bonif_inputs", clear_on_submit=False):
        b1, b2, b3 = st.columns(3)
        with b1:
            bon_hogar = st.number_input(
                "Bonificación por seguro de hogar (%)",
                min_value=0.0, max_value=5.0, value=0.0, step=0.01, format="%.2f", key="bon_hogar"
            )
        with b2:
            bon_vida = st.number_input(
                "Bonificación por seguro de vida (%)",
                min_value=0.0, max_value=5.0, value=0.0, step=0.01, format="%.2f", key="bon_vida"
            )
        with b3:
            bon_otras = st.number_input(
                "Otras bonificaciones (%)",
                min_value=0.0, max_value=10.0, value=0.0, step=0.01, format="%.2f", key="bon_otras"
            )
        _ = st.form_submit_button("🧮 Calcular ahorro")

    bon_total = float(bon_hogar + bon_vida + bon_otras)
    annual_rate_bonif = max(float(annual_rate_pct_b - bon_total), 0.0)

    if bon_total > annual_rate_pct_b:
        st.warning("La bonificación total supera el TIN: el TIN bonificado se ha limitado a 0,00%.")

    r_monthly_bonif = (annual_rate_bonif / 100.0) / 12.0
    df_bon = amortization_schedule(principal_b, r_monthly_bonif, n_months_b)
    monthly_payment_bon = float(df_bon["Cuota"].iloc[0]) if not df_bon.empty else 0.0

    ahorro_cuota_mes = monthly_payment_base - monthly_payment_bon
    ahorro_anual = ahorro_cuota_mes * 12

    st.markdown(
        f"""
        <div style="
            background:#e8f0fe;
            border:1px solid #4A90E2;
            border-radius:12px;
            padding:1rem 1.25rem;
            margin:.5rem 0 1rem 0;
        ">
          <div class="value-title">✅ TIN tras bonificaciones</div>
          <div class="value-big">{annual_rate_bonif:.2f} % <span style="font-size:.9rem;color:#5f6570;font-weight:600">
            (bonificación total: {bon_total:.2f} %)
          </span></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    a1, a2, a3 = st.columns(3)

    a1.metric("💳 Cuota mensual (bonificada)", eur(monthly_payment_bon), delta=eur(-ahorro_cuota_mes))

    with a2:
        st.markdown(
            f"""
            <div style="
                background:#e8f5e9;
                border:1px solid #4caf50;
                border-radius:12px;
                padding:1rem 1.25rem;
                margin:.25rem 0 0 0;
            ">
              <div class="value-title">🧾 Ahorro mensual</div>
              <div class="value-big">{eur(ahorro_cuota_mes)}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    a3.metric("📆 Ahorro anual", eur(ahorro_anual))

    st.caption(
        "Nota: aquí medimos el ahorro como reducción de cuota por la bajada del TIN. "
        "No incluye el coste de los seguros/bonificaciones ni otros gastos/comisiones."
    )

    # -------------------------
    # Prima orientativa (ING)
    # -------------------------
    st.divider()

    st.markdown(
        """
        <div class="param-header">
          <span class="param-chip">Prima orientativa</span>
          <span class="param-subtle">
            Calculo orientativo de prima en base a capital y edad calculado a <strong>03/12/2025</strong> referente a primas de <strong>ING</strong> (Invalidez + Fallecimiento).
          </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    cP1, cP2 = st.columns(2)
    with cP1:
        edad_ing = st.number_input(
            "Edad (años)",
            min_value=0, max_value=99, value=30, step=1, key="edad_ing_prima"
        )
    with cP2:
        capital_ing = st.number_input(
            "Capital a cubrir (€)",
            min_value=0.0, max_value=1000000.0, value=100000.0, step=5000.0, format="%.0f", key="capital_ing_prima"
        )

    if capital_ing > 400000 or edad_ing > 65:
        st.warning("Nota que son calculos oreintativos y pueden no ser acordes a partir de 400.000 euros y edades mayores de 65 años")

    if edad_ing <= 0 or capital_ing <= 0:
        st.info("Introduce una edad y un capital válidos para obtener la prima orientativa.")
    else:
        prima_estimada = prima_orientativa_ing(float(edad_ing), float(capital_ing), PRIMA_ING_DF)

        st.markdown(
            f"""
            <div style="
                background:#e8f0fe;
                border:1px solid #4A90E2;
                border-radius:12px;
                padding:1rem 1.25rem;
                margin:.5rem 0 0.25rem 0;
            ">
              <div class="value-title">🧾 Prima orientativa (mensual)</div>
              <div class="value-big">{eur(prima_estimada)}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.caption(
            "Nota: estimación basada en interpolación lineal por edad y capital sobre la matriz interna. "
            "No contempla condiciones de suscripción, salud, profesión, coberturas adicionales, ni promociones."
        )

# ==========================
# TAB 3: Comparador Fija vs Mixta
# ==========================
with tab_comparador:
    st.markdown(
        """
        <div class="param-header">
          <span class="param-chip">Parámetros — Hipoteca Fija (referencia)</span>
          <span class="param-subtle">Estos definen la hipoteca fija con la que igualaremos intereses.</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.form("params_form_cmp_base", clear_on_submit=False):
        c1b, c2b, c3b, c4b = st.columns([1.2, 1, 1, 1])
        with c1b:
            P_cmp = st.number_input(
                "Importe a financiar (Cantidad solicitada)",
                min_value=1000.0, value=100000.0, step=1000.0, format="%.2f", key="p_cmp"
            )
        with c2b:
            Y_cmp = st.slider("Plazo (años)", min_value=1, max_value=40, value=20, step=1, key="y_cmp")
        with c3b:
            Rfix_cmp = st.number_input(
                "Interés fijo de referencia (% TIN anual)",
                min_value=0.0, max_value=30.0, value=3.0, step=0.05, format="%.2f", key="rfix_cmp"
            )
        with c4b:
            _ = st.selectbox(
                "Mes de inicio (opcional, para agrupación anual)",
                options=list(range(1, 13)), index=0, format_func=lambda m: f"{m:02d}", key="m_cmp"
            )
        _ = st.form_submit_button("✅ Aplicar parámetros de FIJA")

    n_cmp = Y_cmp * 12
    rfix_m = (Rfix_cmp / 100.0) / 12.0

    st.markdown(
        """
        <div class="param-header">
          <span class="param-chip">Parámetros — Hipoteca Mixta</span>
          <span class="param-subtle">Periodo 1 fijo y cálculo del tipo necesario en el periodo 2 (variable).</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.form("params_form_cmp_mixed", clear_on_submit=False):
        c1m, c2m = st.columns([1, 1])
        with c1m:
            Y_change = st.slider(
                "Año donde cambias de fijo a variable",
                min_value=0, max_value=Y_cmp, value=min(5, Y_cmp), step=1, key="y_change"
            )
        with c2m:
            R1_mixed = st.number_input(
                "Interés aplicado en el periodo 1 (fijo de la mixta) — % TIN anual",
                min_value=0.0, max_value=30.0, value=2.5, step=0.05, format="%.2f", key="r1_mixed"
            )
        _ = st.form_submit_button("🧮 Calcular interés necesario del periodo 2 (variable)")

    if P_cmp <= 0 or n_cmp <= 0:
        st.warning("Introduce un importe y un plazo válidos.")
        st.stop()

    df_fix_cmp = amortization_schedule(P_cmp, rfix_m, n_cmp)
    total_interest_fixed = float(df_fix_cmp["Intereses"].sum())
    monthly_payment_fixed = float(df_fix_cmp["Cuota"].iloc[0])

    m1_months = Y_change * 12
    r1_m = (R1_mixed / 100.0) / 12.0
    r2_m_solution, tgt_fixed, mixed_total, i_p1, i_p2 = solve_r2_for_equal_interest(
        P_cmp, n_cmp, rfix_m, r1_m, m1_months
    )

    colA, colB, colC = st.columns(3)
    colA.metric("💡 Intereses totales FIJA (objetivo)", eur(tgt_fixed))
    colB.metric("💳 Cuota mensual FIJA", eur(monthly_payment_fixed))
    colC.metric("⏱️ Meses totales", f"{n_cmp}")

    st.divider()

    n2 = max(n_cmp - m1_months, 0)

    if r1_m == 0:
        cuota_p1 = P_cmp / n_cmp
    else:
        cuota_p1 = P_cmp * r1_m / (1 - (1 + r1_m) ** (-n_cmp))

    if n2 > 0:
        r2_for_calc = r2_m_solution if r2_m_solution is not None else 0.0
        _, _, _, saldo_p1_tmp = mixed_total_interest(P_cmp, n_cmp, r1_m, m1_months, r2_for_calc)
        if r2_for_calc == 0:
            cuota_p2 = saldo_p1_tmp / n2
        else:
            cuota_p2 = saldo_p1_tmp * r2_for_calc / (1 - (1 + r2_for_calc) ** (-n2))
    else:
        cuota_p2 = 0.0

    cI1, cI2, cI3, cI4 = st.columns(4)

    with cI1:
        st.markdown(
            f"<div class='value-title'>Interés periodo 1 (mixta)</div>"
            f"<div class='value-big'>{R1_mixed:.3f} % TIN</div>",
            unsafe_allow_html=True
        )

    with cI2:
        if m1_months >= n_cmp:
            st.markdown(
                "<div class='soft-box'>"
                "<div class='value-title'>Interés necesario periodo 2 (mixta)</div>"
                "<div class='value-big'>—</div>"
                "</div>",
                unsafe_allow_html=True
            )
        else:
            if r2_m_solution is None:
                st.markdown(
                    "<div class='soft-box'>"
                    "<div class='value-title'>Interés necesario periodo 2 (mixta)</div>"
                    "<div class='value-big'>No encontrado</div>"
                    "</div>",
                    unsafe_allow_html=True
                )
            else:
                r2_annual_pct = r2_m_solution * 12 * 100.0
                st.markdown(
                    f"<div class='soft-box'>"
                    f"<div class='value-title'>Interés necesario periodo 2 (mixta)</div>"
                    f"<div class='value-big'>{r2_annual_pct:.3f} % TIN</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

    with cI3:
        st.markdown(
            f"<div class='value-title'>Cuota mensual periodo 1</div>"
            f"<div class='value-big'>{eur(cuota_p1)}</div>",
            unsafe_allow_html=True
        )

    with cI4:
        if n2 > 0 and r2_m_solution is not None:
            st.markdown(
                f"<div class='value-title'>Cuota mensual periodo 2</div>"
                f"<div class='value-big'>{eur(cuota_p2)}</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='value-title'>Cuota mensual periodo 2</div>"
                f"<div class='value-big'>—</div>",
                unsafe_allow_html=True
            )

    r2_for_table = r2_m_solution if r2_m_solution is not None else 0.0
    mixed_total_chk, ip1_chk, ip2_chk, saldo_p1 = mixed_total_interest(
        P_cmp, n_cmp, r1_m, m1_months, r2_for_table
    )

    st.markdown("### 📘 Resumen — Hipoteca Fija")
    fija_df = pd.DataFrame({
        "Concepto": [
            "Cuota mensual a pagar",
            "Valor Hipoteca",
            "Intereses Totales",
            "Suma Capital+Intereses"
        ],
        "Valor": [
            monthly_payment_fixed,
            P_cmp,
            tgt_fixed,
            P_cmp + tgt_fixed
        ]
    })
    st.dataframe(fija_df.style.format({"Valor": eur}), use_container_width=True)

    st.markdown("### 🧩 Resumen — Hipoteca Mixta")
    mixta_df = pd.DataFrame({
        "Concepto": [
            "Cuota a pagar periodo 1",
            "Cuota a pagar periodo 2",
            "Intereses periodo 1",
            "Intereses periodo 2",
            "Valor Hipoteca",
            "Intereses Totales",
            "Suma Capital+Intereses"
        ],
        "Valor": [
            cuota_p1,
            cuota_p2 if r2_m_solution is not None else np.nan,
            ip1_chk,
            ip2_chk if r2_m_solution is not None else np.nan,
            P_cmp,
            mixed_total_chk,
            P_cmp + mixed_total_chk
        ]
    })
    st.dataframe(
        mixta_df.style.format({"Valor": lambda x: "—" if (isinstance(x, float) and np.isnan(x)) else eur(x)}),
        use_container_width=True
    )

    diff = mixed_total_chk - tgt_fixed
    st.caption(f"Diferencia (mixta - fija): {eur(diff)} (≈ 0 si la solución iguala los intereses).")

# =========
# TAB 4: Publicidad (solo imagen local)
# =========
with tab_publicidad:
    st.markdown("<div style='text-align:center'>", unsafe_allow_html=True)
    st.image("publi.jpg", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# =========
# TAB 5: Analiza Inversión
# =========
with tab_inversion:
    st.markdown(
        """
        <div class="param-header">
          <span class="param-chip">Hipoteca de la Inversión</span>
          <span class="param-subtle">Configura el precio, financiación y condiciones para ver tu cuota mensual.</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.form("params_form_inversion", clear_on_submit=False):
        c1, c2 = st.columns([1.2, 1])
        with c1:
            precio_vivienda = st.number_input(
                "Precio de la vivienda (€)",
                min_value=10000.0, value=200000.0, step=1000.0, format="%.2f", key="precio_inv"
            )
        with c2:
            pct_financiacion = st.slider(
                "Porcentaje de financiación (%)",
                min_value=0, max_value=100, value=90, step=5, key="pct_finan_inv"
            )

        c3, c4 = st.columns([1, 1])
        with c3:
            plazo_inv = st.slider(
                "Plazo de la hipoteca (años)",
                min_value=1, max_value=40, value=30, step=1, key="plazo_inv"
            )
        with c4:
            interes_inv = st.number_input(
                "Interés aplicado (% TIN anual)",
                min_value=0.0, max_value=30.0, value=2.7, step=0.05, format="%.2f", key="interes_inv"
            )
        _ = st.form_submit_button("✅ Calcular cuota")

    importe_financiado = precio_vivienda * pct_financiacion / 100
    n_meses_inv = plazo_inv * 12
    r_mensual_inv = (interes_inv / 100.0) / 12.0
    df_inv = amortization_schedule(importe_financiado, r_mensual_inv, n_meses_inv)

    cuota_mensual_inv = 0.0
    if not df_inv.empty:
        cuota_mensual_inv = float(df_inv["Cuota"].iloc[0])

        st.markdown(
            f"""
            <div style="
                background:#e8f0fe;
                border:1px solid #4A90E2;
                border-radius:12px;
                padding:1rem 1.25rem;
                margin:.5rem 0 1rem 0;
            ">
              <div class="value-title">💳 Cuota mensual hipoteca</div>
              <div class="value-big">{eur(cuota_mensual_inv)}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    # --- Apartado 2: Aportación Inicial ---
    st.markdown(
        """
        <div class="param-header">
          <span class="param-chip">Aportación Inicial</span>
          <span class="param-subtle">Entrada + impuestos + gastos fijos + comisión de apertura (+ extra opcional).</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    comunidades = {
        "IVA (Vivienda nueva)": (0.10, 0.012),
        "Andalucía": (0.07, 0.015),
        "Aragón": (0.085, 0.012),
        "Asturias": (0.08, 0.015),
        "Baleares": (0.08, 0.0075),
        "Canarias": (0.065, 0.015),
        "Cantabria": (0.08, 0.015),
        "Castilla León": (0.08, 0.015),
        "Castilla la Mancha": (0.09, 0.015),
        "Cataluña": (0.10, 0.015),
        "Comunidad Valenciana": (0.10, 0.015),
        "Extremadura": (0.08, 0.015),
        "Galicia": (0.10, 0.015),
        "Comunidad de Madrid": (0.06, 0.0075),
        "Murcia": (0.08, 0.015),
        "Navarra": (0.06, 0.005),
        "País Vasco": (0.07, 0.005),
        "La Rioja": (0.07, 0.01)
    }

    comunidad = st.selectbox("Comunidad Autónoma", list(comunidades.keys()), key="comunidad_inv")
    itp, ajd = comunidades[comunidad]

    entrada_pct = 100 - pct_financiacion
    entrada_eur = precio_vivienda * entrada_pct / 100
    impuestos = precio_vivienda * (itp + ajd)

    def _fmt_pct(x: float) -> str:
        s = f"{x:.2f}".rstrip("0").rstrip(".")
        return s.replace(".", ",")

    itp_text = _fmt_pct(itp * 100)
    ajd_text = _fmt_pct(ajd * 100)

    registro_notaria = 1500
    tasacion = 400
    gestoria = 400
    comision_apertura = importe_financiado * 0.02

    aportacion_extra = st.number_input(
        "Aportación extra (reforma / otro concepto) (€)",
        min_value=0.0, value=0.0, step=100.0, format="%.2f", key="aport_extra"
    )

    gastos_fijos = registro_notaria + tasacion + gestoria
    aportacion_total = entrada_eur + impuestos + gastos_fijos + comision_apertura + aportacion_extra

    cA, cB, cC = st.columns(3)
    cA.metric("💰 Entrada", f"{entrada_pct:.1f}% = {eur(entrada_eur)}")

    with cB:
        st.markdown(
            f"""
            <div class='value-title'>
                📑 Impuestos (ITP/IVA + AJD)
                <span style="font-size:0.85em;color:#5f6570;margin-left:.35rem">
                    ITP/IVA {itp_text}% + AJD {ajd_text}%
                </span>
            </div>
            <div class='value-big'>{eur(impuestos)}</div>
            """,
            unsafe_allow_html=True
        )

    cC.metric("🧾 Gastos fijos (Reg.+Not.+Tas.+Gest.)", eur(gastos_fijos))
    st.metric("💸 Comisión de apertura (2%)", eur(comision_apertura))

    st.markdown(
        f"""
        <div style="
            background:#e8f0fe;
            border:1px solid #4A90E2;
            border-radius:12px;
            padding:1rem 1.25rem;
            margin:.5rem 0 1rem 0;
        ">
          <div class="value-title">📊 Aportación inicial total</div>
          <div class="value-big">{eur(aportacion_total)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    resumen_df = pd.DataFrame({
        "Concepto": [
            "Entrada (no financiado)",
            "Impuestos (ITP/IVA + AJD)",
            "Registro y Notaría",
            "Tasación inmueble",
            "Gestoría",
            "Comisión apertura (2%)",
            "Aportación extra (reforma / otros)",
            "TOTAL APORTACIÓN INICIAL"
        ],
        "Importe": [
            entrada_eur,
            impuestos,
            registro_notaria,
            tasacion,
            gestoria,
            comision_apertura,
            aportacion_extra,
            aportacion_total
        ]
    })

    with st.expander("📘 Resumen — Aportación Inicial", expanded=False):
        st.dataframe(resumen_df.style.format({"Importe": eur}), use_container_width=True)

    st.caption(
        "Nota: Gastos fijos asumidos: Registro y Notaría = 1500 €, Tasación = 400 €, Gestoría = 400 €. "
        "La comisión de apertura es el 2% del importe financiado."
    )

    st.divider()

    # --- Apartado 3: Ingresos por Alquiler — Cashflow ---
    st.markdown(
        """
        <div class="param-header">
          <span class="param-chip">Ingresos de Alquiler</span>
          <span class="param-subtle">Define ingresos y gastos para calcular el cashflow anual.</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.form("params_form_alquiler", clear_on_submit=False):
        c1, c2 = st.columns([1, 1])
        with c1:
            alquiler_mensual = st.number_input(
                "Alquiler mensual estimado (€)",
                min_value=0.0, value=1000.0, step=50.0, format="%.2f", key="alq_mensual"
            )
            comunidad_mensual = st.number_input(
                "Comunidad (mensual) (€)",
                min_value=0.0, value=40.0, step=5.0, format="%.2f", key="comunidad_mensual"
            )
            seguros_mensual = st.number_input(
                "Seguros (mensual) (€)",
                min_value=0.0, value=60.0, step=10.0, format="%.2f", key="seguros_mensual"
            )
        with c2:
            ibi_anual = st.number_input(
                "IBI (anual) (€)",
                min_value=0.0, value=150.0, step=25.0, format="%.2f", key="ibi_anual"
            )
            mantenimiento_anual = st.number_input(
                "Mantenimiento (anual) (€)",
                min_value=0.0, value=0.0, step=50.0, format="%.2f", key="mnt_anual"
            )
        _ = st.form_submit_button("✅ Calcular cashflow")

    ingresos_anuales = alquiler_mensual * 12
    hipoteca_anual = float(cuota_mensual_inv) * 12

    otros_gastos_anuales = (
        ibi_anual +
        comunidad_mensual * 12 +
        mantenimiento_anual +
        seguros_mensual * 12
    )
    gastos_anuales_totales = otros_gastos_anuales + hipoteca_anual
    cashflow_anual = ingresos_anuales - gastos_anuales_totales

    cA, cB, cC = st.columns(3)
    cA.metric("📈 Ingresos anuales por alquiler", eur(ingresos_anuales))
    cB.metric("🏦 Gastos de hipoteca anuales", eur(hipoteca_anual))
    cC.metric("📉 Otros gastos anuales (IBI + comunidad + mantenimiento + seguros)", eur(otros_gastos_anuales))

    st.markdown(
        f"""
        <div style="
            background:#e8f0fe;
            border:1px solid #4A90E2;
            border-radius:12px;
            padding:1rem 1.25rem;
            margin:.5rem 0 1rem 0;
        ">
          <div class="value-title">💧 Cashflow anual</div>
          <div class="value-big">{eur(cashflow_anual)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption("El cashflow anual mostrado **incluye** hipoteca. No incluye vacancias, IRPF ni otros posibles ajustes.")

    st.divider()

    # --- Apartado 4: Rentabilidad (ratios) ---
    st.markdown("<h2 style='margin:0 0 .5rem 0'>📈 Rentabilidad</h2>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="param-subtle" style="margin-bottom:.5rem">
          Ratios clave de la inversión. Valores destacados en verde y una breve descripción bajo cada uno.
        </div>
        """,
        unsafe_allow_html=True
    )

    default_horizonte = int(plazo_inv)
    horizonte_anios = st.number_input(
        "Horizonte (años) para comparar el interés compuesto",
        min_value=1, max_value=40, value=default_horizonte, step=1, key="horizonte_comp"
    )

    r_simple = 0.0 if (aportacion_total <= 0) else (cashflow_anual / aportacion_total)
    r_comp = ((1 + horizonte_anios * r_simple) ** (1 / horizonte_anios)) - 1

    def fmt_pct(x: float) -> str:
        return f"{x*100:,.2f} %".replace(".", ",").replace(",", "X").replace("X", ".")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            f"""
            <div style="
                background:#e8f5e9;
                border:1px solid #4caf50;
                border-radius:12px;
                padding:1rem 1.25rem;
                margin:.5rem 0 1rem 0;
            ">
              <div class="value-title">💶 Rentabilidad sobre aportación (Cash-on-Cash)</div>
              <div class="value-big">{fmt_pct(r_simple)}</div>
              <div style="font-size:0.9em;color:#5f6570;margin-top:.35rem">
                <em>Cashflow anual / Aportación inicial</em>. También llamado <strong>Cash-on-Cash Return (CoC)</strong>.
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""
            <div style="
                background:#e8f5e9;
                border:1px solid #4caf50;
                border-radius:12px;
                padding:1rem 1.25rem;
                margin:.5rem 0 1rem 0;
            ">
              <div class="value-title">📈 Interés compuesto equivalente</div>
              <div class="value-big">{fmt_pct(r_comp)}</div>
              <div style="font-size:0.9em;color:#5f6570;margin-top:.35rem">
                Tasa anual constante que, durante {int(horizonte_anios)} año(s), genera el mismo beneficio que una
                rentabilidad simple de {fmt_pct(r_simple)}. (Fórmula: <em>((1 + n·r)<sup>1/n</sup> − 1)</em>).<br/>
                También conocida como <strong>Tasa Anual Equivalente (TAE) de la inversión</strong>.
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    n_h = int(horizonte_anios)
    years_list = list(range(1, n_h + 1))

    def comp_equiv(r: float, n: int) -> float:
        base = 1 + n * r
        return (base ** (1 / n) - 1) if base > 0 else np.nan

    df_ratios = pd.DataFrame({
        "Año": years_list,
        "Rentabilidad sobre aportación (Cash-on-Cash)": [r_simple] * n_h,
        "Interés compuesto equivalente": [comp_equiv(r_simple, n) for n in years_list],
    })

    df_display = df_ratios.copy()
    df_display["Rentabilidad sobre aportación (Cash-on-Cash)"] = df_display["Rentabilidad sobre aportación (Cash-on-Cash)"].map(fmt_pct)
    df_display["Interés compuesto equivalente"] = df_display["Interés compuesto equivalente"].map(fmt_pct)

    with st.expander("🔍 Comparativa por año (simple vs compuesto)", expanded=False):
        st.dataframe(df_display, use_container_width=True, hide_index=True)
