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

@st.cache_data(show_spinner=False)
def baixar_dolar(inicio):
    """
    Baixa o histórico do dólar USD/BRL via Yahoo Finance.
    Ticker usado: BRL=X
    """
    dados = yf.download(
        tickers="BRL=X",
        start=inicio,
        auto_adjust=True,
        progress=False
    )

    if dados.empty:
        return pd.DataFrame()

    if isinstance(dados.columns, pd.MultiIndex):
        dados = dados["Close"]
    else:
        dados = dados[["Close"]]

    dados.columns = ["USD_BRL"]

    return dados.dropna()


def converter_retorno_percentual(valor):
    """
    Converte retornos em formato texto, percentual ou número decimal.
    Aceita:
    - '5,98%'
    - '5.98%'
    - 0.0598
    - 5.98
    """
    if pd.isna(valor):
        return np.nan

    if isinstance(valor, str):
        valor = valor.strip().replace("%", "").replace(",", ".")

        try:
            valor = float(valor)
        except Exception:
            return np.nan

    if abs(valor) > 1:
        valor = valor / 100

    return valor


def tratar_upload_rf(arquivo):
    """
    Trata upload da rentabilidade histórica da RF offshore.
    Espera uma planilha com pelo menos duas colunas:
    - Data
    - Retorno
    """
    if arquivo is None:
        return pd.DataFrame()

    try:
        if arquivo.name.endswith(".csv"):
            df = pd.read_csv(arquivo)
        else:
            df = pd.read_excel(arquivo)
    except Exception as e:
        st.error(f"Erro ao ler arquivo de RF: {e}")
        return pd.DataFrame()

    df.columns = [str(col).strip().lower() for col in df.columns]

    coluna_data = None
    coluna_retorno = None

    for col in df.columns:
        if "data" in col or "mês" in col or "mes" in col:
            coluna_data = col

        if "retorno" in col or "rentabilidade" in col or "rent" in col:
            coluna_retorno = col

    if coluna_data is None or coluna_retorno is None:
        st.error("A planilha precisa ter uma coluna de Data e uma coluna de Retorno/Rentabilidade.")
        return pd.DataFrame()

    df = df[[coluna_data, coluna_retorno]].copy()
    df.columns = ["Data", "RF"]

    df["Data"] = pd.to_datetime(df["Data"], errors="coerce")
    df["RF"] = df["RF"].apply(converter_retorno_percentual)

    df = df.dropna()

    df = df.sort_values("Data")

    df = df.set_index("Data")

    return df


def transformar_diario_para_mensal(precos):
    """
    Converte preços diários em retornos mensais.
    """
    precos_mensais = precos.resample("ME").last()
    retornos_mensais = precos_mensais.pct_change().dropna()

    return retornos_mensais


def calcular_drawdown(serie_retorno):
    """
    Calcula drawdown máximo de uma série de retornos.
    """
    acumulado = (1 + serie_retorno).cumprod()
    pico = acumulado.cummax()
    drawdown = acumulado / pico - 1

    return drawdown.min()


def calcular_var_historico(serie_retorno, nivel=0.05):
    """
    Calcula VaR histórico.
    """
    return serie_retorno.quantile(nivel)


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
# MOTOR DE RISCO: RF + DÓLAR + ETFs
# =========================

st.divider()

st.header("2. Motor de Risco da Carteira")

st.markdown(
    """
    Nesta etapa, o app calcula a volatilidade histórica dos principais componentes da carteira:

    - RF offshore;
    - ETFs internacionais;
    - Dólar USD/BRL;
    - ETFs convertidos para BRL.
    """
)

# =========================
# DÓLAR HISTÓRICO
# =========================

st.subheader("💵 Dólar Histórico")

with st.spinner("Baixando histórico do dólar..."):
    dados_dolar = baixar_dolar(data_inicio_historico.strftime("%Y-%m-%d"))

if dados_dolar.empty:
    st.warning("Não foi possível baixar o histórico do dólar via yfinance.")
    retornos_dolar_mensal = pd.DataFrame()
else:
    retornos_dolar_mensal = transformar_diario_para_mensal(dados_dolar)
    retornos_dolar_mensal.columns = ["Dólar"]

    vol_dolar_anual = retornos_dolar_mensal["Dólar"].std() * np.sqrt(12)
    retorno_dolar_anual = retornos_dolar_mensal["Dólar"].mean() * 12

    col_d1, col_d2 = st.columns(2)

    with col_d1:
        st.metric("Retorno Médio Anual do Dólar", f"{retorno_dolar_anual:.2%}")

    with col_d2:
        st.metric("Volatilidade Anual do Dólar", f"{vol_dolar_anual:.2%}")

    fig_dolar = go.Figure()

    perf_dolar = (1 + retornos_dolar_mensal["Dólar"]).cumprod() - 1

    fig_dolar.add_trace(
        go.Scatter(
            x=perf_dolar.index,
            y=perf_dolar,
            mode="lines",
            name="USD/BRL"
        )
    )

    fig_dolar.update_layout(
        title="Retorno Acumulado do Dólar",
        xaxis_title="Data",
        yaxis_title="Retorno Acumulado",
        yaxis_tickformat=".0%"
    )

    st.plotly_chart(fig_dolar, use_container_width=True)


# =========================
# UPLOAD DA RF OFFSHORE
# =========================

st.subheader("🏦 Rentabilidade Histórica da RF Offshore")

arquivo_rf = st.file_uploader(
    "Faça upload da planilha com a rentabilidade histórica da RF offshore",
    type=["xlsx", "xls", "csv"]
)

moeda_rf = st.selectbox(
    "A rentabilidade da RF está em qual moeda?",
    ["BRL", "USD"],
    index=0
)

rf_historica = tratar_upload_rf(arquivo_rf)

if rf_historica.empty:
    st.info(
        """
        Nenhum histórico de RF foi carregado ainda.

        Para ativar o cálculo completo de risco consolidado, envie uma planilha com:

        - Data
        - Retorno
        """
    )
else:
    st.success("Histórico de RF carregado com sucesso.")

    st.dataframe(
        rf_historica.tail(12).style.format({
            "RF": "{:.2%}"
        }),
        use_container_width=True
    )

    # Caso a RF esteja em USD, convertemos para BRL usando dólar
    if moeda_rf == "USD" and not retornos_dolar_mensal.empty:
        rf_historica = rf_historica.join(retornos_dolar_mensal, how="inner")
        rf_historica["RF_BRL"] = (1 + rf_historica["RF"]) * (1 + rf_historica["Dólar"]) - 1
        serie_rf_final = rf_historica["RF_BRL"].rename("RF")
    else:
        serie_rf_final = rf_historica["RF"].rename("RF")

    vol_rf_anual = serie_rf_final.std() * np.sqrt(12)
    retorno_rf_anual = serie_rf_final.mean() * 12
    drawdown_rf = calcular_drawdown(serie_rf_final)
    var_rf = calcular_var_historico(serie_rf_final)

    col_rf1, col_rf2, col_rf3, col_rf4 = st.columns(4)

    with col_rf1:
        st.metric("Retorno Médio Anual RF", f"{retorno_rf_anual:.2%}")

    with col_rf2:
        st.metric("Volatilidade Anual RF", f"{vol_rf_anual:.2%}")

    with col_rf3:
        st.metric("Drawdown Máximo RF", f"{drawdown_rf:.2%}")

    with col_rf4:
        st.metric("VaR Mensal 95% RF", f"{var_rf:.2%}")


# =========================
# ETFs EM BASE MENSAL E EM BRL
# =========================

st.subheader("🌎 ETFs em USD e Convertidos para BRL")

retornos_etfs_mensal_usd = transformar_diario_para_mensal(dados_etfs)

# Carteira de ETFs igualmente ponderada por enquanto
retorno_etf_usd = retornos_etfs_mensal_usd.mean(axis=1).rename("ETFs_USD")

if not retornos_dolar_mensal.empty:
    base_etf_brl = pd.concat(
        [retorno_etf_usd, retornos_dolar_mensal["Dólar"]],
        axis=1
    ).dropna()

    base_etf_brl["ETFs_BRL"] = (
        (1 + base_etf_brl["ETFs_USD"]) *
        (1 + base_etf_brl["Dólar"]) - 1
    )

    serie_etf_brl = base_etf_brl["ETFs_BRL"]

    vol_etf_usd_anual = retorno_etf_usd.std() * np.sqrt(12)
    vol_etf_brl_anual = serie_etf_brl.std() * np.sqrt(12)

    retorno_etf_usd_anual = retorno_etf_usd.mean() * 12
    retorno_etf_brl_anual = serie_etf_brl.mean() * 12

    col_etf1, col_etf2, col_etf3, col_etf4 = st.columns(4)

    with col_etf1:
        st.metric("Retorno Anual ETFs USD", f"{retorno_etf_usd_anual:.2%}")

    with col_etf2:
        st.metric("Vol Anual ETFs USD", f"{vol_etf_usd_anual:.2%}")

    with col_etf3:
        st.metric("Retorno Anual ETFs BRL", f"{retorno_etf_brl_anual:.2%}")

    with col_etf4:
        st.metric("Vol Anual ETFs BRL", f"{vol_etf_brl_anual:.2%}")

else:
    serie_etf_brl = pd.Series(dtype=float)
    st.warning("Sem dólar histórico, não foi possível converter ETFs para BRL.")       

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
# VOLATILIDADE CONSOLIDADA DA CARTEIRA
# =========================

st.subheader("📊 Volatilidade Consolidada da Carteira")

if not rf_historica.empty and not serie_etf_brl.empty:

    base_risco = pd.concat(
        [
            serie_rf_final.rename("RF"),
            serie_etf_brl.rename("ETFs_BRL")
        ],
        axis=1
    ).dropna()

    if len(base_risco) < 6:
        st.warning("Ainda há poucos dados em comum para calcular a volatilidade consolidada.")
    else:
        matriz_correlacao = base_risco.corr()
        matriz_cov = base_risco.cov() * 12

        pesos_carteira_final = np.array([
            meta_rf / 100,
            meta_rv / 100
        ])

        vol_carteira_final = np.sqrt(
            np.dot(
                pesos_carteira_final.T,
                np.dot(matriz_cov.values, pesos_carteira_final)
            )
        )

        retorno_esperado_final = (
            pesos_carteira_final[0] * base_risco["RF"].mean() * 12 +
            pesos_carteira_final[1] * base_risco["ETFs_BRL"].mean() * 12
        )

        sharpe_simples = retorno_esperado_final / vol_carteira_final if vol_carteira_final != 0 else np.nan

        col_v1, col_v2, col_v3 = st.columns(3)

        with col_v1:
            st.metric("Retorno Esperado Carteira 80/20", f"{retorno_esperado_final:.2%}")

        with col_v2:
            st.metric("Volatilidade Carteira 80/20", f"{vol_carteira_final:.2%}")

        with col_v3:
            st.metric("Sharpe Simples", f"{sharpe_simples:.2f}")

        st.markdown("### Correlação RF x ETFs em BRL")

        fig_corr_consolidada = px.imshow(
            matriz_correlacao,
            text_auto=".2f",
            aspect="auto",
            title="Matriz de Correlação Consolidada"
        )

        st.plotly_chart(fig_corr_consolidada, use_container_width=True)

        # =========================
        # EVOLUÇÃO DA VOLATILIDADE PELO GLIDE PATH
        # =========================

        st.markdown("### Evolução da Volatilidade Projetada pelo Glide Path")

        historico_mensal_df["Volatilidade Projetada"] = np.nan

        for idx, linha in historico_mensal_df.iterrows():

            peso_rv = linha["% ETFs"]
            peso_rf = linha["% RF"]

            pesos_dinamicos = np.array([peso_rf, peso_rv])

            vol_dinamica = np.sqrt(
                np.dot(
                    pesos_dinamicos.T,
                    np.dot(matriz_cov.values, pesos_dinamicos)
                )
            )

            historico_mensal_df.loc[idx, "Volatilidade Projetada"] = vol_dinamica

        fig_vol = go.Figure()

        fig_vol.add_trace(
            go.Scatter(
                x=historico_mensal_df["Data"],
                y=historico_mensal_df["Volatilidade Projetada"] * 100,
                mode="lines+markers",
                name="Volatilidade Projetada"
            )
        )

        fig_vol.update_layout(
            title="Evolução da Volatilidade Anualizada Projetada",
            xaxis_title="Data",
            yaxis_title="Volatilidade Anualizada (%)"
        )

        st.plotly_chart(fig_vol, use_container_width=True)

        st.dataframe(
            historico_mensal_df[
                [
                    "Data",
                    "% ETFs",
                    "% RF",
                    "Volatilidade Projetada"
                ]
            ].style.format({
                "% ETFs": "{:.2%}",
                "% RF": "{:.2%}",
                "Volatilidade Projetada": "{:.2%}"
            }),
            use_container_width=True
        )

else:
    st.info(
        """
        Para calcular a volatilidade consolidada, envie a rentabilidade histórica da RF offshore.

        O app já consegue calcular:
        - Volatilidade dos ETFs;
        - Volatilidade do dólar;
        - ETFs convertidos para BRL.

        Mas a carteira consolidada precisa da série histórica da RF.
        """
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