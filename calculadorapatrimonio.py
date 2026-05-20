# =========================
# IMPORTAÇÕES
# =========================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go

from datetime import date, timedelta


# =========================
# CONFIGURAÇÃO DA PÁGINA
# =========================

st.set_page_config(
    page_title="M Wealth - Simulador Quantitativo",
    layout="wide"
)

# =========================
# IDENTIDADE VISUAL
# =========================

COR_PRINCIPAL = "#131925"

st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: {COR_PRINCIPAL};
            color: #FFFFFF;
        }}

        section[data-testid="stSidebar"] {{
            background-color: #0E131D;
        }}

        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span {{
            color: #FFFFFF !important;
        }}

        h1, h2, h3, h4 {{
            color: #FFFFFF;
        }}

        .stMetric {{
            background-color: #1C2433;
            padding: 16px;
            border-radius: 14px;
            border: 1px solid rgba(255,255,255,0.08);
        }}

        div[data-testid="stMetricValue"] {{
            color: #FFFFFF;
        }}

        div[data-testid="stMetricLabel"] {{
            color: #D8DCE3;
        }}

        .stDataFrame {{
            border-radius: 12px;
        }}

        div[data-testid="stTabs"] button {{
            color: #FFFFFF;
        }}

        div[data-testid="stTabs"] button[aria-selected="true"] {{
            border-bottom: 3px solid #FFFFFF;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# CABEÇALHO
# =========================

col_logo, col_titulo = st.columns([1, 5])

with col_logo:
    st.image("Logo-M-Wealth.png", width=250)

with col_titulo:
    st.title("Simulador Patrimonial Internacional")
    st.markdown(
        """
        Ferramenta para planejar, de forma gradual e controlada, a exposição internacional em ETFs.
        """
    )


# =========================
# FUNÇÕES AUXILIARES
# =========================

def formatar_moeda(valor):
    try:
        return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return valor


def calcular_meses(data_inicial, data_final):
    meses = (data_final.year - data_inicial.year) * 12 + (data_final.month - data_inicial.month)

    if data_final.day >= data_inicial.day:
        meses += 1

    return max(meses, 1)


@st.cache_data(show_spinner=False)
def baixar_precos(tickers, inicio):
    if isinstance(tickers, str):
        tickers = [tickers]

    tickers = list(dict.fromkeys([t.strip().upper() for t in tickers if t.strip() != ""]))

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

    if isinstance(dados.columns, pd.MultiIndex):
        if "Close" in dados.columns.get_level_values(0):
            dados = dados["Close"]
        else:
            return pd.DataFrame()
    else:
        if "Close" in dados.columns:
            dados = dados[["Close"]]
            dados.columns = tickers[:1]
        else:
            return pd.DataFrame()

    if isinstance(dados, pd.Series):
        dados = dados.to_frame(name=tickers[0])

    dados = dados.dropna(axis=1, how="all")

    return dados


def transformar_diario_para_mensal(precos):
    precos_mensais = precos.resample("ME").last()
    retornos_mensais = precos_mensais.pct_change().dropna()
    return retornos_mensais


def calcular_metricas_mensais(serie):
    retorno_anual = serie.mean() * 12
    vol_anual = serie.std() * np.sqrt(12)

    acumulado = (1 + serie).cumprod()
    pico = acumulado.cummax()
    drawdown = acumulado / pico - 1
    max_drawdown = drawdown.min()

    var_95 = serie.quantile(0.05)

    return retorno_anual, vol_anual, max_drawdown, var_95


def gerar_fluxo_mensal(data_inicial, data_final, aporte_2026, aporte_2027, aporte_2028, aporte_padrao):
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


def calcular_glide_path(
    patrimonio_inicial,
    fluxo_df,
    meta_rv,
    meses
):
    total_aportes = fluxo_df["Aporte Mensal"].sum()
    patrimonio_final_sem_rentabilidade = patrimonio_inicial + total_aportes

    meta_rv_financeira = patrimonio_final_sem_rentabilidade * (meta_rv / 100)
    meta_rf_financeira = patrimonio_final_sem_rentabilidade * (1 - meta_rv / 100)

    rv_atual = 0.0
    rf_atual = patrimonio_inicial

    necessidade_total_etfs = max(meta_rv_financeira - rv_atual, 0)
    aporte_mensal_medio_etfs = necessidade_total_etfs / meses

    aporte_mensal_medio_total = total_aportes / meses
    aporte_mensal_medio_rf = max(aporte_mensal_medio_total - aporte_mensal_medio_etfs, 0)

    rv_acumulada = rv_atual
    rf_acumulada = rf_atual

    historico = []

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

        historico.append({
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

    historico_df = pd.DataFrame(historico)

    resumo = {
        "total_aportes": total_aportes,
        "patrimonio_final_sem_rentabilidade": patrimonio_final_sem_rentabilidade,
        "meta_rv_financeira": meta_rv_financeira,
        "meta_rf_financeira": meta_rf_financeira,
        "aporte_mensal_medio_etfs": aporte_mensal_medio_etfs,
        "aporte_mensal_medio_rf": aporte_mensal_medio_rf
    }

    return historico_df, resumo


def simular_cenario(historico_df, retorno_rf_aa, retorno_etf_aa):
    rf_mensal = (1 + retorno_rf_aa) ** (1 / 12) - 1
    etf_mensal = (1 + retorno_etf_aa) ** (1 / 12) - 1

    saldo_rf = 0
    saldo_etf = 0

    # Como o patrimônio inicial já está em RF
    if len(historico_df) > 0:
        saldo_rf = historico_df.iloc[0]["Saldo RF"] - historico_df.iloc[0]["Aporte em RF"]
        saldo_etf = 0

    serie = []

    for _, linha in historico_df.iterrows():
        saldo_rf += linha["Aporte em RF"]
        saldo_etf += linha["Aporte em ETFs"]

        saldo_rf *= (1 + rf_mensal)
        saldo_etf *= (1 + etf_mensal)

        patrimonio = saldo_rf + saldo_etf

        serie.append({
            "Data": linha["Data"],
            "RF": saldo_rf,
            "ETFs": saldo_etf,
            "Patrimônio": patrimonio,
            "% ETFs": saldo_etf / patrimonio if patrimonio > 0 else 0
        })

    return pd.DataFrame(serie)


def rodar_monte_carlo(historico_df, media_mensal, cov_mensal, n_simulacoes=1000):
    resultados = []

    for sim in range(n_simulacoes):
        saldo_rf = 0
        saldo_etf = 0

        if len(historico_df) > 0:
            saldo_rf = historico_df.iloc[0]["Saldo RF"] - historico_df.iloc[0]["Aporte em RF"]

        patrimonio_path = []

        for _, linha in historico_df.iterrows():
            saldo_rf += linha["Aporte em RF"]
            saldo_etf += linha["Aporte em ETFs"]

            retornos_simulados = np.random.multivariate_normal(
                mean=media_mensal,
                cov=cov_mensal
            )

            retorno_rf = retornos_simulados[0]
            retorno_etf = retornos_simulados[1]

            saldo_rf *= (1 + retorno_rf)
            saldo_etf *= (1 + retorno_etf)

            patrimonio = saldo_rf + saldo_etf
            patrimonio_path.append(patrimonio)

        resultados.append(patrimonio_path)

    resultados_df = pd.DataFrame(resultados).T
    resultados_df.index = historico_df["Data"].values

    percentis = pd.DataFrame({
        "P5": resultados_df.quantile(0.05, axis=1),
        "P25": resultados_df.quantile(0.25, axis=1),
        "P50": resultados_df.quantile(0.50, axis=1),
        "P75": resultados_df.quantile(0.75, axis=1),
        "P95": resultados_df.quantile(0.95, axis=1),
    })

    return percentis, resultados_df


def calcular_stress(peso_rf, peso_etf, cenarios):
    linhas = []

    for nome, c in cenarios.items():
        rf_brl = (1 + c["RF_USD"]) * (1 + c["Dolar"]) - 1
        etf_brl = (1 + c["ETF_USD"]) * (1 + c["Dolar"]) - 1

        retorno_carteira = peso_rf * rf_brl + peso_etf * etf_brl

        linhas.append({
            "Cenário": nome,
            "RF USD": c["RF_USD"],
            "ETF USD": c["ETF_USD"],
            "Dólar": c["Dolar"],
            "RF em BRL": rf_brl,
            "ETF em BRL": etf_brl,
            "Impacto Carteira": retorno_carteira
        })

    return pd.DataFrame(linhas)


# =========================
# TÍTULO
# =========================

#st.title("M Wealth - Simulador Quantitativo")

#st.markdown(
#    """
#    Simulador institucional para construção gradual de exposição internacional em ETFs,
#    com plano de evolução da carteira, cenários, risco histórico, stress test e Monte Carlo.
#    """
#)


# =========================
# SIDEBAR
# =========================

st.sidebar.header("Configurações")

hoje = date.today()

def moeda_para_float(valor):
    """
    Converte texto em padrão brasileiro para float.
    Ex:
    R$ 4.098.000,00 -> 4098000.00
    """
    if isinstance(valor, (int, float)):
        return float(valor)

    valor = str(valor)
    valor = valor.replace("R$", "")
    valor = valor.replace(" ", "")
    valor = valor.replace(".", "")
    valor = valor.replace(",", ".")

    try:
        return float(valor)
    except Exception:
        return 0.0


def float_para_moeda(valor):
    """
    Converte float para padrão brasileiro.
    """
    return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

patrimonio_inicial_txt = st.sidebar.text_input(
    "Patrimônio inicial",
    value="R$ 4.098.000,00"
)

patrimonio_inicial = moeda_para_float(patrimonio_inicial_txt)

st.sidebar.subheader("Horizonte")

data_final = st.sidebar.date_input(
    "Data Final",
    value=date(2028, 9, 30)
)

if data_final <= hoje:
    st.sidebar.error("A data final precisa ser posterior à data de hoje.")
    st.stop()

meses = calcular_meses(hoje, data_final)

st.sidebar.write(f"Meses até a meta: **{meses}**")

st.sidebar.subheader("Meta")

meta_rv = st.sidebar.slider(
    "Meta Final em ETFs (%)",
    min_value=0,
    max_value=100,
    value=20,
    step=1
)

meta_rf = 100 - meta_rv

st.sidebar.write(f"Meta RF: **{meta_rf}%**")

st.sidebar.subheader("Aportes")

aporte_2026_txt = st.sidebar.text_input(
    "Aporte mensal em 2026",
    value="R$ 150.000,00"
)

aporte_2027_txt = st.sidebar.text_input(
    "Aporte mensal em 2027",
    value="R$ 400.000,00"
)

aporte_2028_txt = st.sidebar.text_input(
    "Aporte mensal em 2028",
    value="R$ 500.000,00"
)

aporte_padrao_txt = st.sidebar.text_input(
    "Aporte mensal após 2028",
    value="R$ 500.000,00"
)

aporte_2026 = moeda_para_float(aporte_2026_txt)
aporte_2027 = moeda_para_float(aporte_2027_txt)
aporte_2028 = moeda_para_float(aporte_2028_txt)
aporte_padrao = moeda_para_float(aporte_padrao_txt)

st.sidebar.subheader("Proxy de RF Offshore")

rf_proxy = st.sidebar.selectbox(
    "ETF usado como proxy da RF",
    ["SHY", "IEF", "SGOV", "BIL", "TLT"],
    index=1
)

st.sidebar.caption(
    "SHY = Treasury curto | IEF = Treasury intermediário | TLT = Treasury longo | SGOV/BIL = T-Bills"
)

st.sidebar.subheader("ETFs")

etfs_manuais = st.sidebar.text_input(
    "ETFs da carteira",
    value="NOBL",
    placeholder="Ex: NOBL, VOO, SCHD, ACWI"
)

etfs = [
    ticker.strip().upper()
    for ticker in etfs_manuais.replace(";", ",").split(",")
    if ticker.strip() != ""
]

etfs = list(dict.fromkeys(etfs))

st.sidebar.subheader("Histórico")

anos_historico = st.sidebar.slider(
    "Anos de histórico",
    min_value=1,
    max_value=10,
    value=3,
    step=1
)

data_inicio_historico = hoje - timedelta(days=365 * anos_historico)

st.sidebar.write(f"Início do histórico: **{data_inicio_historico.strftime('%d/%m/%Y')}**")

n_simulacoes = st.sidebar.slider(
    "Simulações Monte Carlo",
    min_value=500,
    max_value=5000,
    value=1000,
    step=500
)


# =========================
# DADOS
# =========================

if len(etfs) == 0:
    st.warning("Selecione ou insira pelo menos um ETF.")
    st.stop()

tickers_para_baixar = list(dict.fromkeys(etfs + [rf_proxy, "BRL=X"]))

with st.spinner("Baixando dados históricos..."):
    precos = baixar_precos(tickers_para_baixar, data_inicio_historico.strftime("%Y-%m-%d"))

if precos.empty:
    st.error("Não foi possível baixar os dados históricos. Verifique os tickers.")
    st.stop()

ativos_validos = list(precos.columns)

if "BRL=X" not in ativos_validos:
    st.error("Não foi possível puxar o dólar BRL=X.")
    st.stop()

if rf_proxy not in ativos_validos:
    st.error(f"Não foi possível puxar o proxy de RF: {rf_proxy}.")
    st.stop()

etfs_validos = [ticker for ticker in etfs if ticker in ativos_validos]

if len(etfs_validos) == 0:
    st.error("Nenhum ETF selecionado retornou dados válidos.")
    st.stop()

retornos_mensais = transformar_diario_para_mensal(precos)

retorno_dolar = retornos_mensais["BRL=X"].rename("Dólar")
retorno_rf_usd = retornos_mensais[rf_proxy].rename("RF_USD")

retornos_etfs_usd = retornos_mensais[etfs_validos]
retorno_etf_usd = retornos_etfs_usd.mean(axis=1).rename("ETFs_USD")

base_risco = pd.concat(
    [retorno_rf_usd, retorno_etf_usd, retorno_dolar],
    axis=1
).dropna()

base_risco["RF_BRL"] = (1 + base_risco["RF_USD"]) * (1 + base_risco["Dólar"]) - 1
base_risco["ETFs_BRL"] = (1 + base_risco["ETFs_USD"]) * (1 + base_risco["Dólar"]) - 1

base_carteira = base_risco[["RF_BRL", "ETFs_BRL"]].dropna()

media_mensal = base_carteira.mean().values
cov_mensal = base_carteira.cov().values

ret_rf_aa, vol_rf_aa, dd_rf, var_rf = calcular_metricas_mensais(base_carteira["RF_BRL"])
ret_etf_aa, vol_etf_aa, dd_etf, var_etf = calcular_metricas_mensais(base_carteira["ETFs_BRL"])

correlacao_rf_etf = base_carteira.corr().iloc[0, 1]


# =========================
# Plano de evolução da carteira
# =========================

fluxo_df = gerar_fluxo_mensal(
    data_inicial=hoje,
    data_final=data_final,
    aporte_2026=aporte_2026,
    aporte_2027=aporte_2027,
    aporte_2028=aporte_2028,
    aporte_padrao=aporte_padrao
)

historico_mensal_df, resumo = calcular_glide_path(
    patrimonio_inicial=patrimonio_inicial,
    fluxo_df=fluxo_df,
    meta_rv=meta_rv,
    meses=meses
)

matriz_cov_anual = base_carteira.cov() * 12
pesos_finais = np.array([meta_rf / 100, meta_rv / 100])

vol_final = np.sqrt(
    np.dot(
        pesos_finais.T,
        np.dot(matriz_cov_anual.values, pesos_finais)
    )
)

retorno_final = (
    pesos_finais[0] * ret_rf_aa +
    pesos_finais[1] * ret_etf_aa
)

sharpe_simples = retorno_final / vol_final if vol_final != 0 else np.nan


# =========================
# EVOLUÇÃO DA VOLATILIDADE PELO Plano de evolução da carteira
# =========================

historico_mensal_df["Volatilidade Projetada"] = np.nan

for idx, linha in historico_mensal_df.iterrows():

    peso_rf = linha["% RF"]
    peso_etf = linha["% ETFs"]

    pesos_dinamicos = np.array([peso_rf, peso_etf])

    vol_dinamica = np.sqrt(
        np.dot(
            pesos_dinamicos.T,
            np.dot(matriz_cov_anual.values, pesos_dinamicos)
        )
    )

    historico_mensal_df.loc[idx, "Volatilidade Projetada"] = vol_dinamica


# =========================
# CENÁRIOS
# =========================

cenario_conservador = simular_cenario(
    historico_mensal_df,
    retorno_rf_aa=max(ret_rf_aa - 0.02, -0.10),
    retorno_etf_aa=max(ret_etf_aa - 0.06, -0.20)
)

cenario_base = simular_cenario(
    historico_mensal_df,
    retorno_rf_aa=ret_rf_aa,
    retorno_etf_aa=ret_etf_aa
)

cenario_otimista = simular_cenario(
    historico_mensal_df,
    retorno_rf_aa=ret_rf_aa + 0.02,
    retorno_etf_aa=ret_etf_aa + 0.06
)

cenarios_df = pd.DataFrame({
    "Cenário": ["Conservador", "Base", "Otimista"],
    "Retorno RF a.a.": [max(ret_rf_aa - 0.02, -0.10), ret_rf_aa, ret_rf_aa + 0.02],
    "Retorno ETFs a.a.": [max(ret_etf_aa - 0.06, -0.20), ret_etf_aa, ret_etf_aa + 0.06],
    "Patrimônio Final": [
        cenario_conservador.iloc[-1]["Patrimônio"],
        cenario_base.iloc[-1]["Patrimônio"],
        cenario_otimista.iloc[-1]["Patrimônio"]
    ],
    "% Final ETFs": [
        cenario_conservador.iloc[-1]["% ETFs"],
        cenario_base.iloc[-1]["% ETFs"],
        cenario_otimista.iloc[-1]["% ETFs"]
    ]
})


# =========================
# MONTE CARLO
# =========================

percentis_mc, resultados_mc = rodar_monte_carlo(
    historico_df=historico_mensal_df,
    media_mensal=media_mensal,
    cov_mensal=cov_mensal,
    n_simulacoes=n_simulacoes
)

# =========================
# STRESS TEST
# =========================

cenarios_stress = {
    "Crise Global": {
        "RF_USD": 0.04,
        "ETF_USD": -0.25,
        "Dolar": 0.15
    },
    "Brasil Risk-Off": {
        "RF_USD": 0.02,
        "ETF_USD": -0.08,
        "Dolar": 0.20
    },
    "Abertura de Juros EUA": {
        "RF_USD": -0.07,
        "ETF_USD": -0.10,
        "Dolar": 0.08
    },
    "Soft Landing": {
        "RF_USD": 0.04,
        "ETF_USD": 0.12,
        "Dolar": -0.05
    }
}

stress_df = calcular_stress(
    peso_rf=meta_rf / 100,
    peso_etf=meta_rv / 100,
    cenarios=cenarios_stress
)


# =========================
# ABAS
# =========================

aba_resumo, aba_ativos, aba_simulacoes, aba_avancado = st.tabs(
    [
        "Resumo e Plano",
        "Ativos e Risco",
        "Cenários e Simulações",
        "Avançado"
    ]
)

# =========================
# ABA 1 — RESUMO E PLANO
# =========================

with aba_resumo:

    st.header("Resumo Executivo e Plano de Aportes")

    st.markdown(
        """
        Esta aba mostra a trajetória necessária para que a carteira alcance a alocação-alvo
        dentro do prazo definido, considerando os aportes mensais informados.
        """
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Patrimônio Inicial", formatar_moeda(patrimonio_inicial))

    with col2:
        st.metric("Patrimônio Final sem Rentabilidade", formatar_moeda(resumo["patrimonio_final_sem_rentabilidade"]))

    with col3:
        st.metric("Meta Financeira em ETFs", formatar_moeda(resumo["meta_rv_financeira"]))

    with col4:
        st.metric("Meses até a Meta", meses)

    col5, col6, col7, col8 = st.columns(4)

    with col5:
        st.metric("Aporte Médio em ETFs", formatar_moeda(resumo["aporte_mensal_medio_etfs"]))

    with col6:
        st.metric("Aporte Médio em RF", formatar_moeda(resumo["aporte_mensal_medio_rf"]))

    with col7:
        st.metric(f"Vol Final {meta_rf}/{meta_rv}", f"{vol_final:.2%}")

    with col8:
        st.metric("Sharpe Simples", f"{sharpe_simples:.2f}")

    st.subheader("Convergência para a Meta")

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
        annotation_text="Meta Final",
        annotation_position="top left"
    )

    fig_glide.update_layout(
        title="Evolução da Exposição em ETFs",
        xaxis_title="Data",
        yaxis_title="% em ETFs"
    )

    st.plotly_chart(fig_glide, use_container_width=True)

    st.subheader("Evolução da Volatilidade da Carteira")

    st.markdown(
        """
        A volatilidade projetada considera a mudança gradual dos pesos entre RF e ETFs
        ao longo do tempo, usando o histórico do proxy de RF e da carteira de ETFs em BRL.
        """
    )

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
        title="Volatilidade Anualizada Projetada pelo Plano de evolução da carteira",
        xaxis_title="Data",
        yaxis_title="Volatilidade Anualizada (%)"
    )

    st.plotly_chart(fig_vol, use_container_width=True)

    st.subheader("Plano Mensal de Aportes")

    st.dataframe(
        historico_mensal_df.style.format({
            "Aporte Total": "R$ {:,.2f}",
            "Aporte em ETFs": "R$ {:,.2f}",
            "Aporte em RF": "R$ {:,.2f}",
            "Saldo ETFs": "R$ {:,.2f}",
            "Saldo RF": "R$ {:,.2f}",
            "Patrimônio Total": "R$ {:,.2f}",
            "% ETFs": "{:.2%}",
            "% RF": "{:.2%}",
            "Volatilidade Projetada": "{:.2%}"
        }),
        use_container_width=True
    )

    st.subheader("Distribuição Mensal dos Aportes")

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
# ABA 2 — ATIVOS E RISCO
# =========================

with aba_ativos:

    st.header("Ativos e Risco Histórico")

    st.subheader("Ativos utilizados")

    col_a1, col_a2, col_a3 = st.columns(3)

    with col_a1:
        st.metric("Proxy RF", rf_proxy)

    with col_a2:
        st.metric("ETFs válidos", len(etfs_validos))

    with col_a3:
        st.metric("Correlação RF x ETFs", f"{correlacao_rf_etf:.2f}")

    st.write("ETFs considerados:", etfs_validos)

    metricas_risco = pd.DataFrame({
        "Classe": ["RF Proxy em BRL", "ETFs em BRL"],
        "Retorno Anual": [ret_rf_aa, ret_etf_aa],
        "Volatilidade Anual": [vol_rf_aa, vol_etf_aa],
        "Drawdown Máximo": [dd_rf, dd_etf],
        "VaR Mensal 95%": [var_rf, var_etf]
    })

    st.dataframe(
        metricas_risco.style.format({
            "Retorno Anual": "{:.2%}",
            "Volatilidade Anual": "{:.2%}",
            "Drawdown Máximo": "{:.2%}",
            "VaR Mensal 95%": "{:.2%}"
        }),
        use_container_width=True
    )

    st.subheader("Correlação")

    fig_corr = px.imshow(
        base_carteira.corr(),
        text_auto=".2f",
        aspect="auto",
        title="Correlação RF Proxy x ETFs em BRL"
    )

    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("Performance Histórica em BRL")

    perf = (1 + base_carteira).cumprod() - 1

    fig_perf = go.Figure()

    for col in perf.columns:
        fig_perf.add_trace(
            go.Scatter(
                x=perf.index,
                y=perf[col],
                mode="lines",
                name=col
            )
        )

    fig_perf.update_layout(
        title="Retorno Acumulado — RF Proxy e ETFs em BRL",
        yaxis_tickformat=".0%"
    )

    st.plotly_chart(fig_perf, use_container_width=True)


# =========================
# ABA 3 — CENÁRIOS E SIMULAÇÕES
# =========================

with aba_simulacoes:

    st.header("Cenários e Simulações")

    st.markdown(
        """
        Esta aba concentra as análises prospectivas da carteira.  
        O objetivo é visualizar diferentes possibilidades de evolução patrimonial,
        desde cenários determinísticos até simulações probabilísticas e choques de mercado.
        """
    )

    # =========================
    # CENÁRIOS
    # =========================

    st.subheader("1. Cenários Patrimoniais")

    st.markdown(
        """
        Os cenários patrimoniais usam hipóteses anuais de retorno para RF e ETFs.
        Eles ajudam a visualizar uma trajetória conservadora, uma trajetória base
        e uma trajetória otimista até a data final do estudo.
        """
    )

    st.dataframe(
        cenarios_df.style.format({
            "Retorno RF a.a.": "{:.2%}",
            "Retorno ETFs a.a.": "{:.2%}",
            "Patrimônio Final": "R$ {:,.2f}",
            "% Final ETFs": "{:.2%}"
        }),
        use_container_width=True
    )

    fig_cenarios = go.Figure()

    fig_cenarios.add_trace(
        go.Scatter(
            x=cenario_conservador["Data"],
            y=cenario_conservador["Patrimônio"],
            mode="lines",
            name="Conservador"
        )
    )

    fig_cenarios.add_trace(
        go.Scatter(
            x=cenario_base["Data"],
            y=cenario_base["Patrimônio"],
            mode="lines",
            name="Base"
        )
    )

    fig_cenarios.add_trace(
        go.Scatter(
            x=cenario_otimista["Data"],
            y=cenario_otimista["Patrimônio"],
            mode="lines",
            name="Otimista"
        )
    )

    fig_cenarios.update_layout(
        title="Evolução Patrimonial por Cenário",
        xaxis_title="Data",
        yaxis_title="Patrimônio"
    )

    st.plotly_chart(fig_cenarios, use_container_width=True)

    st.divider()

    # =========================
    # MONTE CARLO
    # =========================

    st.subheader("2. Monte Carlo")

    st.markdown(
        """
        O Monte Carlo simula milhares de trajetórias possíveis para o patrimônio,
        usando a média, volatilidade e covariância histórica mensal entre o proxy de RF
        e a carteira de ETFs, ambos convertidos para BRL.

        O resultado mostra uma faixa probabilística de patrimônio futuro.
        """
    )

    col_mc1, col_mc2, col_mc3 = st.columns(3)

    with col_mc1:
        st.metric("Simulações", n_simulacoes)

    with col_mc2:
        st.metric("P50 Final", formatar_moeda(percentis_mc.iloc[-1]["P50"]))

    with col_mc3:
        st.metric("P5 Final", formatar_moeda(percentis_mc.iloc[-1]["P5"]))

    fig_mc = go.Figure()

    for coluna in ["P95", "P75", "P50", "P25", "P5"]:
        fig_mc.add_trace(
            go.Scatter(
                x=percentis_mc.index,
                y=percentis_mc[coluna],
                mode="lines",
                name=coluna
            )
        )

    fig_mc.update_layout(
        title="Cone Probabilístico — Monte Carlo",
        xaxis_title="Data",
        yaxis_title="Patrimônio"
    )

    st.plotly_chart(fig_mc, use_container_width=True)

    percentis_tabela = percentis_mc.tail(12).copy()
    percentis_tabela.index = pd.to_datetime(percentis_tabela.index).strftime("%d/%m/%Y")
    
    st.dataframe(
        percentis_tabela.style.format("R$ {:,.2f}"),
        use_container_width=True
    )
    st.divider()

    # =========================
    # STRESS TEST
    # =========================

    st.subheader("3. Stress Test")

    st.markdown(
        """
        O stress test aplica choques simultâneos em três componentes:

        - RF em dólar;
        - ETFs em dólar;
        - câmbio USD/BRL.

        A análise mostra como a carteira alvo reagiria em cenários extremos,
        já convertidos para BRL.
        """
    )

    st.dataframe(
        stress_df.style.format({
            "RF USD": "{:.2%}",
            "ETF USD": "{:.2%}",
            "Dólar": "{:.2%}",
            "RF em BRL": "{:.2%}",
            "ETF em BRL": "{:.2%}",
            "Impacto Carteira": "{:.2%}"
        }),
        use_container_width=True
    )

    fig_stress = px.bar(
        stress_df,
        x="Cenário",
        y="Impacto Carteira",
        title="Impacto Estimado por Cenário de Stress",
        text_auto=".2%"
    )

    fig_stress.update_layout(
        yaxis_tickformat=".0%"
    )

    st.plotly_chart(fig_stress, use_container_width=True)


# =========================
# ABA 7 — AVANÇADO
# =========================

with aba_avancado:

    st.header("Avançado")

    st.markdown(
        """
        Esta aba concentra as bases técnicas usadas nos cálculos.
        Ela não precisa ser usada em reunião, mas ajuda na validação do modelo,
        auditoria dos dados e conferência das premissas quantitativas.
        """
    )

    st.subheader("Base de risco mensal")

    st.dataframe(
        base_carteira.tail(24).style.format("{:.2%}"),
        use_container_width=True
    )

    st.subheader("Matriz de covariância mensal")

    st.dataframe(
        base_carteira.cov().style.format("{:.6f}"),
        use_container_width=True
    )

    st.subheader("Matriz de covariância anualizada")

    st.dataframe(
        matriz_cov_anual.style.format("{:.6f}"),
        use_container_width=True
    )

    st.subheader("ETFs individuais — retornos mensais")

    st.dataframe(
        retornos_etfs_usd.tail(24).style.format("{:.2%}"),
        use_container_width=True
    )


st.success("App carregado com sucesso.")