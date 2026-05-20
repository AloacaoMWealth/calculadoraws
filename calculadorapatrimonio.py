# =========================
# IMPORTAÇÕES
# =========================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go

from datetime import datetime, date


# =========================
# CONFIGURAÇÃO DA PÁGINA
# =========================

st.set_page_config(
    page_title="M Wealth - Simulador Quantitativo",
    layout="wide"
)


# =========================
# FUNÇÕES AUXILIARES
# =========================

def formatar_moeda(valor):
    """
    Formata valores em padrão brasileiro.
    """
    try:
        return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return valor


def calcular_meses(data_inicial, data_final):
    """
    Calcula a quantidade de meses entre duas datas.
    """
    meses = (data_final.year - data_inicial.year) * 12 + (data_final.month - data_inicial.month)

    if data_final.day >= data_inicial.day:
        meses += 1

    return max(meses, 1)


@st.cache_data(show_spinner=False)
def baixar_dados(tickers, inicio):
    """
    Baixa dados históricos dos ativos via yfinance.
    Usa Close com auto_adjust=True para evitar erro de Adj Close.
    Trata tanto um ticker quanto múltiplos tickers.
    """

    if isinstance(tickers, str):
        tickers = [tickers]

    tickers = list(dict.fromkeys([ticker.strip().upper() for ticker in tickers if ticker.strip() != ""]))

    if len(tickers) == 0:
        return pd.DataFrame()

    dados = yf.download(
        tickers=tickers,
        start=inicio,
        auto_adjust=True,
        progress=False
    )

    if dados.empty:
        return pd.DataFrame()

    # Caso venha MultiIndex, comum quando há múltiplos ativos
    if isinstance(dados.columns, pd.MultiIndex):
        if "Close" in dados.columns.get_level_values(0):
            dados = dados["Close"]
        else:
            return pd.DataFrame()

    # Caso venha DataFrame simples
    else:
        if "Close" in dados.columns:
            dados = dados[["Close"]]
            dados.columns = tickers[:1]
        else:
            return pd.DataFrame()

    # Garante DataFrame mesmo para um único ativo
    if isinstance(dados, pd.Series):
        dados = dados.to_frame(name=tickers[0])

    # Remove colunas completamente vazias
    dados = dados.dropna(axis=1, how="all")

    return dados


def calcular_metricas(retornos):
    """
    Calcula retorno anualizado e volatilidade anualizada.
    """
    retorno_anual = retornos.mean() * 252
    volatilidade_anual = retornos.std() * np.sqrt(252)

    metricas = pd.DataFrame({
        "Retorno Anual": retorno_anual,
        "Volatilidade Anual": volatilidade_anual
    })

    metricas.index.name = "Ticker"

    return metricas


def gerar_fluxo_mensal(data_inicial, data_final, aporte_2026, aporte_2027, aporte_2028, aporte_padrao):
    """
    Gera uma tabela mensal de aportes de acordo com o ano.
    """
    datas = pd.date_range(start=data_inicial, end=data_final, freq="ME")

    if len(datas) == 0:
        datas = pd.date_range(start=data_inicial, periods=1, freq="ME")

    fluxo = []

    for data_ref in datas:
        ano = data_ref.year

        if ano == 2026:
            aporte = aporte_2026
        elif ano == 2027:
            aporte = aporte_2027
        elif ano == 2028:
            aporte = aporte_2028
        else:
            aporte = aporte_padrao

        fluxo.append({
            "Data": data_ref,
            "Ano": ano,
            "Aporte Mensal": aporte
        })

    return pd.DataFrame(fluxo)


# =========================
# TÍTULO
# =========================

st.title("📊 M Wealth - Simulador Quantitativo Internacional")

st.markdown(
    """
    Plataforma de modelagem patrimonial internacional com foco em:

    - Alocação quantitativa;
    - ETFs internacionais;
    - Simulação patrimonial;
    - Volatilidade consolidada;
    - Construção de exposição gradual em renda variável;
    - Glide path de alocação.
    """
)


# =========================
# SIDEBAR - CONFIGURAÇÕES
# =========================

st.sidebar.header("⚙️ Configurações")


# =========================
# PATRIMÔNIO
# =========================

patrimonio_inicial = st.sidebar.number_input(
    "Patrimônio Inicial (R$)",
    value=4098000.0,
    step=100000.0,
    min_value=0.0
)


# =========================
# HORIZONTE
# =========================

st.sidebar.subheader("📅 Horizonte do Estudo")

data_inicial = st.sidebar.date_input(
    "Data Inicial",
    value=date(2026, 5, 1)
)

data_final = st.sidebar.date_input(
    "Data Final",
    value=date(2028, 9, 30)
)

if data_final <= data_inicial:
    st.sidebar.error("A data final precisa ser posterior à data inicial.")

meses = calcular_meses(data_inicial, data_final)

st.sidebar.info(f"Horizonte calculado: {meses} meses")


# =========================
# META DE ALOCAÇÃO
# =========================

st.sidebar.subheader("🎯 Meta de Alocação")

meta_rv = st.sidebar.slider(
    "Meta Final de Renda Variável / ETFs (%)",
    min_value=0,
    max_value=100,
    value=20,
    step=1
)

meta_rf = 100 - meta_rv

st.sidebar.write(f"Meta RF: **{meta_rf}%**")


# =========================
# APORTES
# =========================

st.sidebar.subheader("💰 Fluxos de Aporte")

aporte_2026 = st.sidebar.number_input(
    "Aporte Mensal 2026",
    value=150000.0,
    step=50000.0,
    min_value=0.0
)

aporte_2027 = st.sidebar.number_input(
    "Aporte Mensal 2027",
    value=400000.0,
    step=50000.0,
    min_value=0.0
)

aporte_2028 = st.sidebar.number_input(
    "Aporte Mensal 2028",
    value=500000.0,
    step=50000.0,
    min_value=0.0
)

aporte_padrao = st.sidebar.number_input(
    "Aporte Mensal para outros anos",
    value=500000.0,
    step=50000.0,
    min_value=0.0
)


# =========================
# ETFs DINÂMICOS
# =========================

st.sidebar.subheader("🌎 ETFs")

lista_base_etfs = [
    "NOBL",
    "VOO",
    "IVV",
    "SPY",
    "SCHD",
    "QUAL",
    "VXUS",
    "VEA",
    "VWO",
    "QQQ",
    "GLD",
    "IAU",
    "TLT",
    "IEF",
    "SHY",
    "BND"
]

etfs_selecionados = st.sidebar.multiselect(
    "Selecione ETFs da lista",
    lista_base_etfs,
    default=["NOBL"]
)

etfs_manuais = st.sidebar.text_input(
    "Inserir ETFs manualmente",
    placeholder="Ex: VTI, ACWI, DGRO"
)

lista_etfs_manuais = []

if etfs_manuais:
    lista_etfs_manuais = [
        ticker.strip().upper()
        for ticker in etfs_manuais.replace(";", ",").split(",")
        if ticker.strip() != ""
    ]

etfs = list(dict.fromkeys(etfs_selecionados + lista_etfs_manuais))

st.sidebar.write("ETFs usados no estudo:")
st.sidebar.write(etfs)


# =========================
# DATA INICIAL HISTÓRICA
# =========================

st.sidebar.subheader("📊 Histórico dos Ativos")

data_inicio_historico = st.sidebar.date_input(
    "Início do histórico dos ETFs",
    value=date(2022, 1, 1)
)


# =========================
# DOWNLOAD E MÉTRICAS DOS ETFs
# =========================

st.divider()

st.header("1. Dados Históricos dos ETFs")

if len(etfs) == 0:
    st.warning("Selecione ou insira pelo menos um ETF para iniciar o estudo.")
    st.stop()

with st.spinner("Baixando dados históricos dos ETFs..."):
    dados_etfs = baixar_dados(etfs, data_inicio_historico.strftime("%Y-%m-%d"))

if dados_etfs.empty:
    st.error(
        "Não foi possível baixar dados dos ETFs informados. "
        "Verifique se os tickers estão corretos no Yahoo Finance."
    )
    st.stop()

# Remove ativos sem dados suficientes
dados_etfs = dados_etfs.dropna(axis=1, how="all")

if dados_etfs.shape[1] == 0:
    st.error("Nenhum dos ETFs retornou dados válidos.")
    st.stop()

retornos = dados_etfs.pct_change().dropna()

if retornos.empty:
    st.error("Não há retornos suficientes para calcular as métricas.")
    st.stop()

metricas = calcular_metricas(retornos)

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.metric("Quantidade de ETFs", len(dados_etfs.columns))

with col_b:
    st.metric("Início do Histórico", dados_etfs.index.min().strftime("%d/%m/%Y"))

with col_c:
    st.metric("Última Data", dados_etfs.index.max().strftime("%d/%m/%Y"))


# =========================
# TABELA DE MÉTRICAS
# =========================

st.subheader("📈 Métricas dos ETFs")

st.dataframe(
    metricas.style.format({
        "Retorno Anual": "{:.2%}",
        "Volatilidade Anual": "{:.2%}"
    }),
    use_container_width=True
)


# =========================
# CORRELAÇÃO
# =========================

st.subheader("🔗 Matriz de Correlação")

correlacao = retornos.corr()

if len(correlacao.columns) >= 2:
    fig_corr = px.imshow(
        correlacao,
        text_auto=".2f",
        aspect="auto",
        title="Correlação entre ETFs"
    )

    st.plotly_chart(fig_corr, use_container_width=True)

else:
    st.info("Com apenas um ETF selecionado, a matriz de correlação será igual a 1.")


# =========================
# PERFORMANCE HISTÓRICA
# =========================

st.subheader("📊 Performance Histórica dos ETFs")

performance = (1 + retornos).cumprod() - 1

fig_perf = go.Figure()

for col in performance.columns:
    fig_perf.add_trace(
        go.Scatter(
            x=performance.index,
            y=performance[col],
            mode="lines",
            name=col
        )
    )

fig_perf.update_layout(
    title="Retorno Acumulado dos ETFs",
    xaxis_title="Data",
    yaxis_title="Retorno Acumulado",
    yaxis_tickformat=".0%"
)

st.plotly_chart(fig_perf, use_container_width=True)


# =========================
# GLIDE PATH FINANCEIRO
# =========================

st.divider()

st.header("2. Glide Path para Alocação Alvo")

fluxo_df = gerar_fluxo_mensal(
    data_inicial=data_inicial,
    data_final=data_final,
    aporte_2026=aporte_2026,
    aporte_2027=aporte_2027,
    aporte_2028=aporte_2028,
    aporte_padrao=aporte_padrao
)

total_aportes = fluxo_df["Aporte Mensal"].sum()
patrimonio_final_sem_rentabilidade = patrimonio_inicial + total_aportes

meta_rv_financeira = patrimonio_final_sem_rentabilidade * (meta_rv / 100)
meta_rf_financeira = patrimonio_final_sem_rentabilidade * (meta_rf / 100)

# Assumindo que hoje a carteira está 100% em RF
rv_atual = 0.0
rf_atual = patrimonio_inicial

necessidade_total_etfs = max(meta_rv_financeira - rv_atual, 0)

aporte_mensal_medio_etfs = necessidade_total_etfs / meses

aporte_mensal_medio_total = total_aportes / meses

aporte_mensal_medio_rf = max(aporte_mensal_medio_total - aporte_mensal_medio_etfs, 0)


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Patrimônio Inicial",
        formatar_moeda(patrimonio_inicial)
    )

with col2:
    st.metric(
        "Total de Aportes",
        formatar_moeda(total_aportes)
    )

with col3:
    st.metric(
        "Patrimônio Final Estimado",
        formatar_moeda(patrimonio_final_sem_rentabilidade)
    )

with col4:
    st.metric(
        "Meses até a Meta",
        meses
    )


col5, col6, col7 = st.columns(3)

with col5:
    st.metric(
        "Meta Financeira em ETFs",
        formatar_moeda(meta_rv_financeira)
    )

with col6:
    st.metric(
        "Aporte Médio Mensal em ETFs",
        formatar_moeda(aporte_mensal_medio_etfs)
    )

with col7:
    st.metric(
        "Aporte Médio Mensal em RF",
        formatar_moeda(aporte_mensal_medio_rf)
    )


# =========================
# EVOLUÇÃO MENSAL DA CARTEIRA
# =========================

st.subheader("📆 Plano Mensal de Aportes")

rv_acumulada = rv_atual
rf_acumulada = rf_atual

historico_mensal = []

for _, linha in fluxo_df.iterrows():

    data_ref = linha["Data"]
    aporte_mensal = linha["Aporte Mensal"]

    saldo_restante_etfs = max(meta_rv_financeira - rv_acumulada, 0)

    aporte_etfs = min(
        aporte_mensal,
        aporte_mensal_medio_etfs,
        saldo_restante_etfs
    )

    aporte_rf = aporte_mensal - aporte_etfs

    rv_acumulada += aporte_etfs
    rf_acumulada += aporte_rf

    patrimonio_total = rv_acumulada + rf_acumulada

    percentual_rv = rv_acumulada / patrimonio_total if patrimonio_total > 0 else 0
    percentual_rf = rf_acumulada / patrimonio_total if patrimonio_total > 0 else 0

    historico_mensal.append({
        "Data": data_ref,
        "Ano": data_ref.year,
        "Aporte Total": aporte_mensal,
        "Aporte em ETFs": aporte_etfs,
        "Aporte em RF": aporte_rf,
        "Saldo ETFs": rv_acumulada,
        "Saldo RF": rf_acumulada,
        "Patrimônio Total": patrimonio_total,
        "% ETFs": percentual_rv,
        "% RF": percentual_rf
    })

historico_mensal_df = pd.DataFrame(historico_mensal)

st.dataframe(
    historico_mensal_df.style.format({
        "Aporte Total": "R$ {:,.2f}",
        "Aporte em ETFs": "R$ {:,.2f}",
        "Aporte em RF": "R$ {:,.2f}",
        "Saldo ETFs": "R$ {:,.2f}",
        "Saldo RF": "R$ {:,.2f}",
        "Patrimônio Total": "R$ {:,.2f}",
        "% ETFs": "{:.2%}",
        "% RF": "{:.2%}"
    }),
    use_container_width=True
)


# =========================
# GRÁFICO DE APORTES
# =========================

st.subheader("💰 Distribuição Mensal dos Aportes")

fig_aportes = go.Figure()

fig_aportes.add_trace(
    go.Bar(
        x=historico_mensal_df["Data"],
        y=historico_mensal_df["Aporte em ETFs"],
        name="Aporte em ETFs"
    )
)

fig_aportes.add_trace(
    go.Bar(
        x=historico_mensal_df["Data"],
        y=historico_mensal_df["Aporte em RF"],
        name="Aporte em RF"
    )
)

fig_aportes.update_layout(
    barmode="stack",
    title="Aportes Mensais por Classe",
    xaxis_title="Data",
    yaxis_title="Valor"
)

st.plotly_chart(fig_aportes, use_container_width=True)


# =========================
# GRÁFICO DE CONVERGÊNCIA
# =========================

st.subheader("🎯 Convergência para Meta de Alocação")

fig_glide = go.Figure()

fig_glide.add_trace(
    go.Scatter(
        x=historico_mensal_df["Data"],
        y=historico_mensal_df["% ETFs"] * 100,
        mode="lines+markers",
        name="% ETFs"
    )
)

fig_glide.add_hline(
    y=meta_rv,
    line_dash="dash",
    annotation_text="Meta Final de ETFs",
    annotation_position="top left"
)

fig_glide.update_layout(
    title="Evolução da Exposição em ETFs",
    xaxis_title="Data",
    yaxis_title="% da Carteira em ETFs"
)

st.plotly_chart(fig_glide, use_container_width=True)


# =========================
# ESTRUTURA PATRIMONIAL FINAL
# =========================

st.subheader("🏦 Estrutura Patrimonial Projetada")

fig_patrimonio = go.Figure()

fig_patrimonio.add_trace(
    go.Bar(
        x=historico_mensal_df["Data"],
        y=historico_mensal_df["Saldo ETFs"],
        name="ETFs"
    )
)

fig_patrimonio.add_trace(
    go.Bar(
        x=historico_mensal_df["Data"],
        y=historico_mensal_df["Saldo RF"],
        name="Renda Fixa"
    )
)

fig_patrimonio.update_layout(
    barmode="stack",
    title="Evolução Patrimonial Projetada",
    xaxis_title="Data",
    yaxis_title="Patrimônio"
)

st.plotly_chart(fig_patrimonio, use_container_width=True)


# =========================
# ALOCAÇÃO ENTRE ETFs
# =========================

st.divider()

st.header("3. Distribuição entre ETFs")

st.markdown(
    """
    Nesta versão inicial, o sistema considera que o valor destinado a ETFs será dividido igualmente entre os ativos selecionados.

    Na próxima etapa, vamos liberar pesos manuais por ETF e também uma otimização quantitativa.
    """
)

peso_por_etf = 1 / len(dados_etfs.columns)

alocacao_etfs = pd.DataFrame({
    "ETF": dados_etfs.columns,
    "Peso no Bloco de ETFs": peso_por_etf,
    "Aporte Médio Mensal Estimado": aporte_mensal_medio_etfs * peso_por_etf,
    "Valor Final Estimado": meta_rv_financeira * peso_por_etf
})

st.dataframe(
    alocacao_etfs.style.format({
        "Peso no Bloco de ETFs": "{:.2%}",
        "Aporte Médio Mensal Estimado": "R$ {:,.2f}",
        "Valor Final Estimado": "R$ {:,.2f}"
    }),
    use_container_width=True
)


# =========================
# PRÓXIMOS MÓDULOS
# =========================

st.divider()

st.header("4. Próximos Módulos Quantitativos")

st.info(
    """
    Próximas implementações:

    1. Upload da rentabilidade histórica da RF offshore;
    2. Dados históricos do dólar;
    3. Volatilidade consolidada RF + ETFs + câmbio;
    4. Matriz de covariância;
    5. Sharpe e drawdown da carteira;
    6. Simulação Monte Carlo;
    7. Stress test;
    8. Otimização dos pesos entre ETFs.
    """
)

st.success("App carregado com sucesso.")