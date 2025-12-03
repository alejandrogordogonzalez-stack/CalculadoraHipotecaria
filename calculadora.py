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
# Utilidades de amortización
# ============================
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
        balance = balance - principal_pay
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
        bal = bal - principal_pay
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

    total_mixed, ip1, ip2, _ = mixed_total_interest(P, n, r1_m, m1, r2_m=lo)
    if f_lo * f_hi > 0:
        return None, target, total_mixed, ip1, ip2

    for _ in range(80):
        mid = (lo + hi) / 2
        f_mid = f(mid)
        if abs(f_mid) < 1e-8:
            lo = hi = mid
            break
        if f_lo * f_mid <= 0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid

    r2_m_solution = (lo + hi) / 2
    total_mixed, ip1, ip2, _ = mixed_total_interest(P, n, r1_m, m1, r2_m_solution)
    return r2_m_solution, target, total_mixed, ip1, ip2

def eur(x: float) -> str:
    return f"{x:,.2f} €"

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
# TAB 1bis: Estudio Bonificaciones (copia de Simulador + bonificaciones)
# =========
with tab_bonif:
    st.markdown(
        """
        <div class="param-header">
          <span class="param-chip">Parámetros</span>
          <span class="param-subtle">Simula tu hipoteca y luego aplica bonificaciones para ver el ahorro en €.</span>
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
            start_month_b = st.selectbox(
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

    total_interest_base = float(df_base["Intereses"].sum())
    monthly_payment_base = float(df_base["Cuota"].iloc[0])

    m1c, m2c, m3c = st.columns(3)
    m1c.metric("💳 Cuota mensual (sin bonificar)", eur(monthly_payment_base))
    m2c.metric("💡 Intereses totales (sin bonificar)", eur(total_interest_base))
    m3c.metric("🗓️ Nº de cuotas (meses)", f"{n_months_b}")

    st.divider()

    st.markdown(
        """
        <div class="param-header">
          <span class="param-chip">Bonificaciones</span>
          <span class="param-subtle">
            Introduce bonificaciones en <strong>puntos porcentuales</strong> sobre el TIN
            (ej.: <strong>0,15</strong> significa <strong>-0,15%</strong>).
          </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.form("params_form_bonif_inputs", clear_on_submit=False):
        b1, b2, b3 = st.columns(3)
        with b1:
            bon_hogar = st.number_input(
                "Bonificación aplicada por seguro de hogar (%)",
                min_value=0.0, max_value=5.0, value=0.0, step=0.01, format="%.2f", key="bon_hogar"
            )
        with b2:
            bon_vida = st.number_input(
                "Bonificación aplicada por seguro de vida (%)",
                min_value=0.0, max_value=5.0, value=0.0, step=0.01, format="%.2f", key="bon_vida"
            )
        with b3:
            bon_otras = st.number_input(
                "Otras bonificaciones (%)",
                min_value=0.0, max_value=10.0, value=0.0, step=0.01, format="%.2f", key="bon_otras"
            )
        _ = st.form_submit_button("🧮 Calcular ahorro con bonificaciones")

    bon_total = float(bon_hogar + bon_vida + bon_otras)
    annual_rate_bonif = max(float(annual_rate_pct_b - bon_total), 0.0)
    r_monthly_bonif = (annual_rate_bonif / 100.0) / 12.0

    df_bon = amortization_schedule(principal_b, r_monthly_bonif, n_months_b)
    total_interest_bon = float(df_bon["Intereses"].sum()) if not df_bon.empty else 0.0
    monthly_payment_bon = float(df_bon["Cuota"].iloc[0]) if not df_bon.empty else 0.0

    ahorro_intereses = total_interest_base - total_interest_bon
    ahorro_cuota_mes = monthly_payment_base - monthly_payment_bon
    ahorro_total_pagado = (principal_b + total_interest_base) - (principal_b + total_interest_bon)

    # Resultados
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
          <div class="value-big">{annual_rate_bonif:.2f} % (bonificación total: {bon_total:.2f} %)</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    a1, a2, a3 = st.columns(3)
    a1.metric("💳 Cuota mensual (bonificada)", eur(monthly_payment_bon), delta=eur(-ahorro_cuota_mes))
    a2.metric("💡 Intereses totales (bonificada)", eur(total_interest_bon), delta=eur(-ahorro_intereses))
    a3.metric("💰 Ahorro total en € (intereses)", eur(ahorro_intereses))

    comp_df = pd.DataFrame({
        "Concepto": [
            "TIN anual",
            "Cuota mensual",
            "Intereses totales",
            "Total pagado (capital + intereses)",
        ],
        "Sin bonificar": [
            f"{annual_rate_pct_b:.2f} %",
            monthly_payment_base,
            total_interest_base,
            principal_b + total_interest_base,
        ],
        "Bonificada": [
            f"{annual_rate_bonif:.2f} %",
            monthly_payment_bon,
            total_interest_bon,
            principal_b + total_interest_bon,
        ],
        "Diferencia (€)": [
            "—",
            ahorro_cuota_mes,
            ahorro_intereses,
            ahorro_total_pagado,
        ]
    })

    st.dataframe(
        comp_df.style.format({
            "Sin bonificar": lambda x: x if isinstance(x, str) else eur(x),
            "Bonificada": lambda x: x if isinstance(x, str) else eur(x),
            "Diferencia (€)": lambda x: x if isinstance(x, str) else eur(x),
        }),
        use_container_width=True
    )

    # Visual rápido: intereses sin vs con
    cmp_bar = go.Figure()
    cmp_bar.add_trace(go.Bar(name="Sin bonificar", x=["Intereses totales"], y=[total_interest_base]))
    cmp_bar.add_trace(go.Bar(name="Bonificada", x=["Intereses totales"], y=[total_interest_bon]))
    cmp_bar.update_layout(
        barmode="group",
        title="Comparativa — Intereses totales",
        yaxis_title="€"
    )
    st.plotly_chart(cmp_bar, use_container_width=True)

    with st.expander("Ver detalle (primeras 12 cuotas) — Sin bonificar vs Bonificada"):
        cL, cR = st.columns(2)
        with cL:
            st.markdown("**Sin bonificar**")
            st.dataframe(df_base.head(12).style.format(
                {"Cuota": eur, "Intereses": eur, "Amortización": eur, "Saldo final": eur}
            ))
        with cR:
            st.markdown("**Bonificada**")
            st.dataframe(df_bon.head(12).style.format(
                {"Cuota": eur, "Intereses": eur, "Amortización": eur, "Saldo final": eur}
            ))

    st.caption("Nota: el ahorro mostrado es puramente financiero por bajada de TIN. No incluye el coste de los seguros/bonificaciones.")

# ==========================
# TAB 2: Comparador Fija vs Mixta
# ==========================
with tab_comparador:
    # --- Parámetros base (FIJA) ---
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
            start_month_cmp = st.selectbox(
                "Mes de inicio (opcional, para agrupación anual)",
                options=list(range(1, 13)), index=0, format_func=lambda m: f"{m:02d}", key="m_cmp"
            )
        _ = st.form_submit_button("✅ Aplicar parámetros de FIJA")

    n_cmp = Y_cmp * 12
    rfix_m = (Rfix_cmp / 100.0) / 12.0

    # --- Parámetros de la MIXTA ---
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

    # --- Cálculo comparador ---
    if P_cmp <= 0 or n_cmp <= 0:
        st.warning("Introduce un importe y un plazo válidos.")
        st.stop()

    # Fija
    df_fix_cmp = amortization_schedule(P_cmp, rfix_m, n_cmp)
    total_interest_fixed = float(df_fix_cmp["Intereses"].sum())
    monthly_payment_fixed = float(df_fix_cmp["Cuota"].iloc[0])

    # Mixta - resolver r2
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

    # Preparar cuotas de mixta
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
    st.dataframe(
        fija_df.style.format({"Valor": eur}),
        use_container_width=True
    )

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
# TAB 3: Publicidad (solo imagen local)
# =========
with tab_publicidad:
    st.markdown("<div style='text-align:center'>", unsafe_allow_html=True)
    st.image("publi.jpg", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# =========
# TAB 4: Analiza Inversión
# =========
with tab_inversion:
    # --- Apartado 1: Hipoteca sobre inversión ---
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
        st.dataframe(
            resumen_df.style.format({"Importe": eur}),
            use_container_width=True
        )

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

    default_horizonte = int(plazo_inv) if 'plazo_inv' in locals() else 5
    horizonte_anios = st.number_input(
        "Horizonte (años) para comparar el interés compuesto",
        min_value=1, max_value=40, value=default_horizonte, step=1, key="horizonte_comp"
    )

    r_simple = 0.0 if (aportacion_total is None or aportacion_total <= 0) else (cashflow_anual / aportacion_total)
    r_comp = ((1 + horizonte_anios * r_simple) ** (1 / horizonte_anios)) - 1

    def fmt_pct(x: float) -> str:
        return f"{x*100:,.2f} %".replace(".", ",")

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
                Tasa anual constante que, durante {horizonte_anios} año(s), genera el mismo beneficio que una
                rentabilidad simple de {fmt_pct(r_simple)}. (Fórmula: <em>((1 + n·r)<sup>1/n</sup> − 1)</em>).<br/>
                También conocida como <strong>Tasa Anual Equivalente (TAE) de la inversión</strong>.
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    n_h = int(horizonte_anios)
    years = list(range(1, n_h + 1))

    def comp_equiv(r: float, n: int) -> float:
        base = 1 + n * r
        return (base ** (1 / n) - 1) if base > 0 else np.nan

    df_ratios = pd.DataFrame({
        "Año": years,
        "Rentabilidad sobre aportación (Cash-on-Cash)": [r_simple] * n_h,
        "Interés compuesto equivalente": [comp_equiv(r_simple, n) for n in years],
    })

    df_display = df_ratios.copy()
    df_display["Rentabilidad sobre aportación (Cash-on-Cash)"] = df_display[
        "Rentabilidad sobre aportación (Cash-on-Cash)"
    ].map(fmt_pct)
    df_display["Interés compuesto equivalente"] = df_display[
        "Interés compuesto equivalente"
    ].map(fmt_pct)

    with st.expander("🔍 Comparativa por año (simple vs compuesto)", expanded=False):
        st.dataframe(df_display, use_container_width=True, hide_index=True)
