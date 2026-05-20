# =========================
# IMPORTAÇÕES
# =========================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go

from datetime import datetime

# =========================
# CONFIGURAÇÃO DA PÁGINA
# =========================

st.set_page_config(
    page_title="M Wealth - Simulador Quantitativo",
    layout="wide"
)

# =========================
# TÍTULO
# =========================

st.title("📊 M Wealth - Simulador Quantitativo Internacional")

st.markdown(
    """
    Plataforma de modelagem patrimonial internacional
    com foco em:

    - Alocação quantitativa;
    - ETFs internacionais;
    - Simulação patrimonial;
    - Volatilidade consolidada;
    - Monte Carlo;
    - Glide path de alocação.
    """
)

# =========================
# SIDEBAR
# =========================

st.sidebar.header("⚙️ Configurações")

# Patrimônio inicial
patrimonio_inicial = st.sidebar.number_input(
    "Patrimônio Inicial (R$)",
    value=4098000.0,
    step=100000.0
)

# Horizonte
ano_final = st.sidebar.selectbox(
    "Ano Final",
    [2027, 2028, 2029, 2030],
    index=1
)

# Meta RV
meta_rv = st.sidebar.slider(
    "Meta Final de Renda Variável (%)",
    0,
    100,
    20
)

meta_rf = 100 - meta_rv

# =========================
# APORTES
# =========================

st.sidebar.subheader("💰 Fluxos de Aporte")

aporte_2026 = st.sidebar.number_input(
    "Aporte Mensal 2026",
    value=150000.0,
    step=50000.0
)

aporte_2027 = st.sidebar.number_input(
    "Aporte Mensal 2027",
    value=400000.0,
    step=50000.0
)

aporte_2028 = st.sidebar.number_input(
    "Aporte Mensal 2028",
    value=500000.0,
    step=50000.0
)

# =========================
# ETFs
# =========================

st.sidebar.subheader("🌎 ETFs")

etfs = st.sidebar.multiselect(
    "Selecione os ETFs",
    ["NOBL", "VOO", "SCHD", "QUAL", "VXUS", "GLD"],
    default=["NOBL"]
)

# =========================
# DOWNLOAD DE DADOS
# =========================

@st.cache_data

def baixar_dados(tickers, inicio="2022-01-01"):

    dados = yf.download(tickers, start=inicio)["Adj Close"]

    return dados

# =========================
# DADOS DOS ETFs
# =========================

if len(etfs) > 0:

    dados_etfs = baixar_dados(etfs)

    retornos = dados_etfs.pct_change().dropna()

    volatilidade_anual = retornos.std() * np.sqrt(252)

    retorno_anual = retornos.mean() * 252

    # =========================
    # MÉTRICAS
    # =========================

    st.subheader("📈 Métricas dos ETFs")

    metricas = pd.DataFrame({
        "Retorno Anual": retorno_anual,
        "Volatilidade Anual": volatilidade_anual
    })

    st.dataframe(metricas.style.format("{:.2%}"))

    # =========================
    # CORRELAÇÃO
    # =========================

    st.subheader("🔗 Correlação")

    correlacao = retornos.corr()

    fig_corr = px.imshow(
        correlacao,
        text_auto=True,
        aspect="auto",
        title="Matriz de Correlação"
    )

    st.plotly_chart(fig_corr, use_container_width=True)

    # =========================
    # EVOLUÇÃO HISTÓRICA
    # =========================

    st.subheader("📊 Evolução Histórica")

    performance = (1 + retornos).cumprod()

    fig_perf = go.Figure()

    for col in performance.columns:

        fig_perf.add_trace(
            go.Scatter(
                x=performance.index,
                y=performance[col],
                mode='lines',
                name=col
            )
        )

    fig_perf.update_layout(
        title="Performance Histórica dos ETFs",
        xaxis_title="Data",
        yaxis_title="Retorno Acumulado"
    )

    st.plotly_chart(fig_perf, use_container_width=True)

# =========================
# SIMULAÇÃO PATRIMONIAL
# =========================

st.subheader("🏦 Simulação Patrimonial")

anos = [2026, 2027, 2028]

aportes = {
    2026: aporte_2026,
    2027: aporte_2027,
    2028: aporte_2028
}

patrimonio = patrimonio_inicial

historico = []

for ano in anos:

    aporte_anual = aportes[ano] * 12

    patrimonio += aporte_anual

    rv = patrimonio * (meta_rv / 100)

    rf = patrimonio * (meta_rf / 100)

    historico.append({
        "Ano": ano,
        "Patrimônio": patrimonio,
        "Renda Variável": rv,
        "Renda Fixa": rf
    })

# =========================
# DATAFRAME
# =========================

historico_df = pd.DataFrame(historico)

st.dataframe(
    historico_df.style.format({
        "Patrimônio": "R$ {:,.2f}",
        "Renda Variável": "R$ {:,.2f}",
        "Renda Fixa": "R$ {:,.2f}"
    })
)

# =========================
# GRÁFICO PATRIMONIAL
# =========================

fig_patrimonio = go.Figure()

fig_patrimonio.add_trace(
    go.Bar(
        x=historico_df["Ano"],
        y=historico_df["Renda Variável"],
        name="Renda Variável"
    )
)

fig_patrimonio.add_trace(
    go.Bar(
        x=historico_df["Ano"],
        y=historico_df["Renda Fixa"],
        name="Renda Fixa"
    )
)

fig_patrimonio.update_layout(
    barmode='stack',
    title="Estrutura Patrimonial Projetada"
)

st.plotly_chart(fig_patrimonio, use_container_width=True)
