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
from io import BytesIO

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    REPORTLAB_DISPONIVEL = True
except Exception:
    REPORTLAB_DISPONIVEL = False


# =========================
# CONFIGURAÇÃO DA PÁGINA
# =========================

st.set_page_config(
    page_title="M Wealth - Simulador Patrimonial Internacional",
    layout="wide"
)


# =========================
# IDENTIDADE VISUAL
# =========================

COR_PRINCIPAL = "#131925"
COR_SIDEBAR = "#0E131D"
COR_CARD = "#1C2433"
COR_INPUT = "#0B1018"
COR_TEXTO = "#FFFFFF"
COR_TEXTO_SECUNDARIO = "#D8DCE3"
COR_DESTAQUE = "#7895E8"

st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: {COR_PRINCIPAL};
            color: {COR_TEXTO};
        }}

        section[data-testid="stSidebar"] {{
            background-color: {COR_SIDEBAR};
        }}

        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span {{
            color: {COR_TEXTO} !important;
        }}

        h1, h2, h3, h4 {{
            color: {COR_TEXTO};
        }}

        p, li, span {{
            color: {COR_TEXTO};
        }}

        .stMetric {{
            background-color: {COR_CARD};
            padding: 16px;
            border-radius: 14px;
            border: 1px solid rgba(255,255,255,0.08);
        }}

        div[data-testid="stMetricValue"] {{
            color: {COR_TEXTO};
        }}

        div[data-testid="stMetricLabel"] {{
            color: {COR_TEXTO_SECUNDARIO};
        }}

        .stDataFrame {{
            border-radius: 12px;
        }}

        div[data-testid="stTabs"] button {{
            color: {COR_TEXTO};
        }}

        div[data-testid="stTabs"] button[aria-selected="true"] {{
            border-bottom: 3px solid {COR_DESTAQUE};
        }}

        input, textarea {{
            background-color: {COR_INPUT} !important;
            color: {COR_TEXTO} !important;
            border-radius: 10px !important;
            border: 1px solid rgba(255,255,255,0.08) !important;
        }}

        div[data-baseweb="select"] > div {{
            background-color: {COR_INPUT} !important;
            color: {COR_TEXTO} !important;
            border-radius: 10px !important;
        }}

        .stSlider > div > div > div {{
            color: {COR_TEXTO} !important;
        }}

        div[data-testid="stExpander"] {{
            background-color: rgba(255,255,255,0.02);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
        }}
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# FUNÇÕES AUXILIARES
# =========================

def moeda_para_float(valor):
    """
    Converte texto em padrão brasileiro para float.
    Ex: R$ 4.098.000,00 -> 4098000.00
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


def formatar_moeda(valor):
    """
    Converte número para padrão brasileiro.
    """
    try:
        return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return valor


def formatar_data_br(valor):
    """
    Formata datas sem hora para padrão brasileiro.
    """
    return pd.to_datetime(valor).strftime("%d/%m/%Y")


def calcular_meses(data_inicial, data_final):
    """
    Calcula quantidade de meses entre duas datas.
    """
    meses = (data_final.year - data_inicial.year) * 12 + (data_final.month - data_inicial.month)

    if data_final.day >= data_inicial.day:
        meses += 1

    return max(meses, 1)


@st.cache_data(show_spinner=False)
def baixar_precos(tickers, inicio):
    """
    Baixa preços ajustados via Yahoo Finance.
    """
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
    """
    Converte preços diários para retornos mensais.
    """
    precos_mensais = precos.resample("ME").last()
    retornos_mensais = precos_mensais.pct_change().dropna()
    return retornos_mensais


def calcular_metricas_mensais(serie):
    """
    Calcula retorno anualizado, oscilação anualizada, maior queda e perda mensal estimada.
    """
    retorno_anual = serie.mean() * 12
    vol_anual = serie.std() * np.sqrt(12)

    acumulado = (1 + serie).cumprod()
    pico = acumulado.cummax()
    drawdown = acumulado / pico - 1
    max_drawdown = drawdown.min()

    var_95 = serie.quantile(0.05)

    return retorno_anual, vol_anual, max_drawdown, var_95


def gerar_fluxo_mensal(data_inicial, data_final, aporte_2026, aporte_2027, aporte_2028, aporte_padrao):
    """
    Gera tabela mensal de aportes por ano.
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


def calcular_plano_evolucao(
    patrimonio_inicial,
    fluxo_df,
    meta_etfs,
    meses
):
    """
    Calcula o plano mensal necessário para alcançar a meta de ETFs.
    Assume patrimônio inicial 100% em Renda Fixa Internacional.
    """
    total_aportes = fluxo_df["Aporte Mensal"].sum()
    patrimonio_final_sem_rentabilidade = patrimonio_inicial + total_aportes

    meta_etfs_financeira = patrimonio_final_sem_rentabilidade * (meta_etfs / 100)
    meta_renda_fixa_financeira = patrimonio_final_sem_rentabilidade * (1 - meta_etfs / 100)

    etfs_atual = 0.0
    renda_fixa_atual = patrimonio_inicial

    necessidade_total_etfs = max(meta_etfs_financeira - etfs_atual, 0)
    aporte_mensal_medio_etfs = necessidade_total_etfs / meses

    aporte_mensal_medio_total = total_aportes / meses
    aporte_mensal_medio_renda_fixa = max(aporte_mensal_medio_total - aporte_mensal_medio_etfs, 0)

    etfs_acumulado = etfs_atual
    renda_fixa_acumulada = renda_fixa_atual

    historico = []

    for _, linha in fluxo_df.iterrows():
        data_ref = linha["Data"]
        aporte_mensal = linha["Aporte Mensal"]

        saldo_restante_etfs = max(meta_etfs_financeira - etfs_acumulado, 0)

        aporte_etfs = min(
            aporte_mensal,
            aporte_mensal_medio_etfs,
            saldo_restante_etfs
        )

        aporte_renda_fixa = aporte_mensal - aporte_etfs

        etfs_acumulado += aporte_etfs
        renda_fixa_acumulada += aporte_renda_fixa

        patrimonio_total = etfs_acumulado + renda_fixa_acumulada

        percentual_etfs = etfs_acumulado / patrimonio_total if patrimonio_total > 0 else 0
        percentual_renda_fixa = renda_fixa_acumulada / patrimonio_total if patrimonio_total > 0 else 0

        historico.append({
            "Data": data_ref,
            "Ano": data_ref.year,
            "Aporte Total": aporte_mensal,
            "Aporte em ETFs": aporte_etfs,
            "Aporte em Renda Fixa": aporte_renda_fixa,
            "Saldo em ETFs": etfs_acumulado,
            "Saldo em Renda Fixa": renda_fixa_acumulada,
            "Patrimônio Total": patrimonio_total,
            "% ETFs": percentual_etfs,
            "% Renda Fixa": percentual_renda_fixa
        })

    historico_df = pd.DataFrame(historico)

    resumo = {
        "total_aportes": total_aportes,
        "patrimonio_final_sem_rentabilidade": patrimonio_final_sem_rentabilidade,
        "meta_etfs_financeira": meta_etfs_financeira,
        "meta_renda_fixa_financeira": meta_renda_fixa_financeira,
        "aporte_mensal_medio_etfs": aporte_mensal_medio_etfs,
        "aporte_mensal_medio_renda_fixa": aporte_mensal_medio_renda_fixa
    }

    return historico_df, resumo


def simular_cenario(historico_df, retorno_renda_fixa_aa, retorno_etfs_aa):
    """
    Simula evolução patrimonial com retornos anuais determinísticos.
    """
    renda_fixa_mensal = (1 + retorno_renda_fixa_aa) ** (1 / 12) - 1
    etfs_mensal = (1 + retorno_etfs_aa) ** (1 / 12) - 1

    saldo_renda_fixa = 0
    saldo_etfs = 0

    if len(historico_df) > 0:
        saldo_renda_fixa = (
            historico_df.iloc[0]["Saldo em Renda Fixa"] -
            historico_df.iloc[0]["Aporte em Renda Fixa"]
        )
        saldo_etfs = 0

    serie = []

    for _, linha in historico_df.iterrows():
        saldo_renda_fixa += linha["Aporte em Renda Fixa"]
        saldo_etfs += linha["Aporte em ETFs"]

        saldo_renda_fixa *= (1 + renda_fixa_mensal)
        saldo_etfs *= (1 + etfs_mensal)

        patrimonio = saldo_renda_fixa + saldo_etfs

        serie.append({
            "Data": linha["Data"],
            "Renda Fixa": saldo_renda_fixa,
            "ETFs": saldo_etfs,
            "Patrimônio": patrimonio,
            "% ETFs": saldo_etfs / patrimonio if patrimonio > 0 else 0
        })

    return pd.DataFrame(serie)


def rodar_simulacao_probabilistica(historico_df, media_mensal, cov_mensal, n_simulacoes=1000):
    """
    Roda simulação probabilística usando média e matriz de covariância mensal.
    """
    resultados = []

    for _ in range(n_simulacoes):
        saldo_renda_fixa = 0
        saldo_etfs = 0

        if len(historico_df) > 0:
            saldo_renda_fixa = (
                historico_df.iloc[0]["Saldo em Renda Fixa"] -
                historico_df.iloc[0]["Aporte em Renda Fixa"]
            )

        patrimonio_path = []

        for _, linha in historico_df.iterrows():
            saldo_renda_fixa += linha["Aporte em Renda Fixa"]
            saldo_etfs += linha["Aporte em ETFs"]

            retornos_simulados = np.random.multivariate_normal(
                mean=media_mensal,
                cov=cov_mensal
            )

            retorno_renda_fixa = retornos_simulados[0]
            retorno_etfs = retornos_simulados[1]

            saldo_renda_fixa *= (1 + retorno_renda_fixa)
            saldo_etfs *= (1 + retorno_etfs)

            patrimonio = saldo_renda_fixa + saldo_etfs
            patrimonio_path.append(patrimonio)

        resultados.append(patrimonio_path)

    resultados_df = pd.DataFrame(resultados).T
    resultados_df.index = historico_df["Data"].values

    percentis = pd.DataFrame({
        "Cenário pessimista": resultados_df.quantile(0.05, axis=1),
        "Faixa inferior": resultados_df.quantile(0.25, axis=1),
        "Cenário central": resultados_df.quantile(0.50, axis=1),
        "Faixa superior": resultados_df.quantile(0.75, axis=1),
        "Cenário otimista": resultados_df.quantile(0.95, axis=1),
    })

    return percentis, resultados_df


def calcular_simulacao_crise(peso_renda_fixa, peso_etfs, cenarios):
    """
    Calcula impacto dos cenários de crise na carteira.
    """
    linhas = []

    for nome, c in cenarios.items():
        renda_fixa_brl = (1 + c["Renda Fixa em USD"]) * (1 + c["Variação do Dólar"]) - 1
        etfs_brl = (1 + c["ETFs em USD"]) * (1 + c["Variação do Dólar"]) - 1

        impacto_carteira = peso_renda_fixa * renda_fixa_brl + peso_etfs * etfs_brl

        linhas.append({
            "Cenário": nome,
            "Renda Fixa em USD": c["Renda Fixa em USD"],
            "ETFs em USD": c["ETFs em USD"],
            "Variação do Dólar": c["Variação do Dólar"],
            "Renda Fixa em BRL": renda_fixa_brl,
            "ETFs em BRL": etfs_brl,
            "Impacto na Carteira": impacto_carteira
        })

    return pd.DataFrame(linhas)


def formatar_datas_tabela(df, coluna_data="Data"):
    """
    Cria cópia do dataframe formatando coluna de data sem hora.
    """
    tabela = df.copy()

    if coluna_data in tabela.columns:
        tabela[coluna_data] = pd.to_datetime(tabela[coluna_data]).dt.strftime("%d/%m/%Y")

    return tabela


def formatar_percentual(valor):
    """
    Formata percentual no padrão brasileiro.
    """
    try:
        return f"{valor:.2%}".replace(".", ",")
    except Exception:
        return valor


def calcular_fronteira_eficiente(retorno_renda_fixa, retorno_etfs, matriz_cov_anual, pontos=101):
    """
    Calcula combinações possíveis entre Renda Fixa Internacional e ETFs.
    Como o modelo tem duas grandes classes, a fronteira é construída variando
    o peso em ETFs de 0% a 100%.
    """
    linhas = []

    for peso_etfs in np.linspace(0, 1, pontos):
        peso_renda_fixa = 1 - peso_etfs
        pesos = np.array([peso_renda_fixa, peso_etfs])

        retorno = peso_renda_fixa * retorno_renda_fixa + peso_etfs * retorno_etfs
        oscilacao = np.sqrt(np.dot(pesos.T, np.dot(matriz_cov_anual.values, pesos)))
        eficiencia = retorno / oscilacao if oscilacao != 0 else np.nan

        linhas.append({
            "% em ETFs": peso_etfs,
            "% em Renda Fixa": peso_renda_fixa,
            "Retorno esperado": retorno,
            "Oscilação esperada": oscilacao,
            "Eficiência risco-retorno": eficiencia
        })

    return pd.DataFrame(linhas)


def gerar_pdf_relatorio(
    patrimonio_inicial,
    data_final,
    meses,
    meta_etfs,
    meta_renda_fixa,
    resumo,
    oscilacao_final,
    eficiencia_risco_retorno,
    referencia_renda_fixa,
    etfs_validos,
    cenarios_df,
    percentis_mc,
    crise_df
):
    """
    Gera um PDF executivo simples para apresentação ao cliente.
    """
    if not REPORTLAB_DISPONIVEL:
        return None

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.3 * cm,
        bottomMargin=1.3 * cm
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="TituloMW",
        parent=styles["Title"],
        textColor=colors.HexColor("#131925"),
        fontSize=18,
        leading=22,
        spaceAfter=14
    ))
    styles.add(ParagraphStyle(
        name="SubtituloMW",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#131925"),
        fontSize=12,
        leading=15,
        spaceBefore=10,
        spaceAfter=8
    ))
    styles.add(ParagraphStyle(
        name="TextoMW",
        parent=styles["BodyText"],
        fontSize=9,
        leading=12
    ))

    elementos = []

    elementos.append(Paragraph("M Wealth - Simulador Patrimonial Internacional", styles["TituloMW"]))
    elementos.append(Paragraph(
        "Relatório executivo de planejamento para construção gradual de exposição internacional em ETFs.",
        styles["TextoMW"]
    ))
    elementos.append(Spacer(1, 8))

    dados_resumo = [
        ["Indicador", "Valor"],
        ["Patrimônio inicial", formatar_moeda(patrimonio_inicial)],
        ["Data final do estudo", pd.to_datetime(data_final).strftime("%d/%m/%Y")],
        ["Meses até a meta", str(meses)],
        ["Meta em ETFs", f"{meta_etfs:.0f}%"],
        ["Meta em Renda Fixa", f"{meta_renda_fixa:.0f}%"],
        ["Patrimônio estimado sem rentabilidade", formatar_moeda(resumo["patrimonio_final_sem_rentabilidade"])],
        ["Meta financeira em ETFs", formatar_moeda(resumo["meta_etfs_financeira"])],
        ["Aporte médio mensal em ETFs", formatar_moeda(resumo["aporte_mensal_medio_etfs"])],
        ["Aporte médio mensal em Renda Fixa", formatar_moeda(resumo["aporte_mensal_medio_renda_fixa"])],
        ["Oscilação esperada da carteira", formatar_percentual(oscilacao_final)],
        ["Eficiência risco-retorno", f"{eficiencia_risco_retorno:.2f}"],
    ]

    elementos.append(Paragraph("Resumo Executivo", styles["SubtituloMW"]))
    tabela_resumo = Table(dados_resumo, colWidths=[8.2 * cm, 7.2 * cm])
    tabela_resumo.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#131925")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D8DCE3")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F7F8FA")),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F5F8")]),
    ]))
    elementos.append(tabela_resumo)
    elementos.append(Spacer(1, 10))

    elementos.append(Paragraph("Ativos utilizados", styles["SubtituloMW"]))
    elementos.append(Paragraph(
        f"Referência de Renda Fixa Internacional: {referencia_renda_fixa}. ETFs considerados: {', '.join(etfs_validos)}.",
        styles["TextoMW"]
    ))

    elementos.append(Paragraph("Cenários patrimoniais", styles["SubtituloMW"]))
    cenarios_tabela = [["Cenário", "Retorno RF a.a.", "Retorno ETFs a.a.", "Patrimônio Final", "% Final em ETFs"]]
    for _, row in cenarios_df.iterrows():
        cenarios_tabela.append([
            row["Cenário"],
            formatar_percentual(row["Retorno Renda Fixa a.a."]),
            formatar_percentual(row["Retorno ETFs a.a."]),
            formatar_moeda(row["Patrimônio Final"]),
            formatar_percentual(row["% Final em ETFs"]),
        ])

    tabela_cenarios = Table(cenarios_tabela, colWidths=[3.2 * cm, 3.1 * cm, 3.1 * cm, 4.0 * cm, 2.8 * cm])
    tabela_cenarios.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#131925")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D8DCE3")),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F5F8")]),
    ]))
    elementos.append(tabela_cenarios)

    elementos.append(Paragraph("Simulação probabilística", styles["SubtituloMW"]))
    elementos.append(Paragraph(
        f"Cenário central ao final do período: {formatar_moeda(percentis_mc.iloc[-1]['Cenário central'])}. "
        f"Cenário pessimista: {formatar_moeda(percentis_mc.iloc[-1]['Cenário pessimista'])}. "
        f"Cenário otimista: {formatar_moeda(percentis_mc.iloc[-1]['Cenário otimista'])}.",
        styles["TextoMW"]
    ))

    elementos.append(Paragraph("Simulação de crise", styles["SubtituloMW"]))
    crise_tabela = [["Cenário", "Impacto na Carteira"]]
    for _, row in crise_df.iterrows():
        crise_tabela.append([row["Cenário"], formatar_percentual(row["Impacto na Carteira"])])

    tabela_crise = Table(crise_tabela, colWidths=[10 * cm, 5.4 * cm])
    tabela_crise.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#131925")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D8DCE3")),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F5F8")]),
    ]))
    elementos.append(tabela_crise)

    elementos.append(Spacer(1, 12))
    elementos.append(Paragraph(
        "Observação: este relatório tem finalidade de estudo e simulação. Os resultados dependem das premissas informadas e dos dados históricos utilizados.",
        styles["TextoMW"]
    ))

    doc.build(elementos)
    buffer.seek(0)
    return buffer


# =========================
# CABEÇALHO
# =========================

st.markdown("<br>", unsafe_allow_html=True)

col_logo, col_titulo = st.columns([1.1, 5])

with col_logo:
    try:
        st.image("Logo-M-Wealth.png", width=210)
    except Exception:
        st.write("M Wealth")

with col_titulo:
    st.title("Simulador Patrimonial Internacional")
    st.markdown(
        """
        Ferramenta para planejamento gradual e controlada de exposição internacional em ETFs.
        """
    )


# =========================
# SIDEBAR
# =========================


hoje = date.today()

patrimonio_inicial_txt = st.sidebar.text_input(
    "Patrimônio inicial",
    value="R$ 4.098.000,00"
)

patrimonio_inicial = moeda_para_float(patrimonio_inicial_txt)

data_final = st.sidebar.date_input(
    "Data final",
    value=date(2028, 9, 30)
)

if data_final <= hoje:
    st.sidebar.error("A data final precisa ser posterior à data de hoje.")
    st.stop()

meses = calcular_meses(hoje, data_final)

st.sidebar.write(f"Meses até a meta: **{meses}**")

meta_etfs = st.sidebar.slider(
    "Meta final em ETFs (%)",
    min_value=0,
    max_value=100,
    value=20,
    step=1
)

meta_renda_fixa = 100 - meta_etfs

st.sidebar.write(f"Meta em Renda Fixa: **{meta_renda_fixa}%**")

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

st.sidebar.subheader("Renda Fixa Internacional")

referencia_renda_fixa = st.sidebar.selectbox(
    "Referência usada para a Renda Fixa",
    ["SHY", "IEF", "SGOV", "BIL", "TLT"],
    index=1
)

st.sidebar.caption(
    "Essa referência é usada apenas para estimar risco e retorno histórico."
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

# Histórico e simulações definidos como padrão para simplificar o uso do app.
ANOS_HISTORICO_PADRAO = 5
N_SIMULACOES_PADRAO = 2000

data_inicio_historico = hoje - timedelta(days=365 * ANOS_HISTORICO_PADRAO)
n_simulacoes = N_SIMULACOES_PADRAO


# =========================
# DADOS
# =========================

if len(etfs) == 0:
    st.warning("Insira pelo menos um ETF.")
    st.stop()

tickers_para_baixar = list(dict.fromkeys(etfs + [referencia_renda_fixa, "BRL=X"]))

with st.spinner("Buscando dados históricos..."):
    precos = baixar_precos(tickers_para_baixar, data_inicio_historico.strftime("%Y-%m-%d"))

if precos.empty:
    st.error("Não foi possível baixar os dados históricos. Verifique os tickers informados.")
    st.stop()

ativos_validos = list(precos.columns)

if "BRL=X" not in ativos_validos:
    st.error("Não foi possível puxar o histórico do dólar.")
    st.stop()

if referencia_renda_fixa not in ativos_validos:
    st.error(f"Não foi possível puxar a referência de Renda Fixa: {referencia_renda_fixa}.")
    st.stop()

etfs_validos = [ticker for ticker in etfs if ticker in ativos_validos]

if len(etfs_validos) == 0:
    st.error("Nenhum ETF informado retornou dados válidos.")
    st.stop()

retornos_mensais = transformar_diario_para_mensal(precos)

retorno_dolar = retornos_mensais["BRL=X"].rename("Dólar")
retorno_renda_fixa_usd = retornos_mensais[referencia_renda_fixa].rename("Renda Fixa USD")

retornos_etfs_usd = retornos_mensais[etfs_validos]
retorno_etfs_usd = retornos_etfs_usd.mean(axis=1).rename("ETFs USD")

base_risco = pd.concat(
    [retorno_renda_fixa_usd, retorno_etfs_usd, retorno_dolar],
    axis=1
).dropna()

base_risco["Renda Fixa em BRL"] = (1 + base_risco["Renda Fixa USD"]) * (1 + base_risco["Dólar"]) - 1
base_risco["ETFs em BRL"] = (1 + base_risco["ETFs USD"]) * (1 + base_risco["Dólar"]) - 1

base_carteira = base_risco[["Renda Fixa em BRL", "ETFs em BRL"]].dropna()

media_mensal = base_carteira.mean().values
cov_mensal = base_carteira.cov().values

ret_renda_fixa_aa, vol_renda_fixa_aa, queda_renda_fixa, perda_ruim_renda_fixa = calcular_metricas_mensais(
    base_carteira["Renda Fixa em BRL"]
)

ret_etfs_aa, vol_etfs_aa, queda_etfs, perda_ruim_etfs = calcular_metricas_mensais(
    base_carteira["ETFs em BRL"]
)

relacao_renda_fixa_etfs = base_carteira.corr().iloc[0, 1]


# =========================
# PLANO DE EVOLUÇÃO DA CARTEIRA
# =========================

fluxo_df = gerar_fluxo_mensal(
    data_inicial=hoje,
    data_final=data_final,
    aporte_2026=aporte_2026,
    aporte_2027=aporte_2027,
    aporte_2028=aporte_2028,
    aporte_padrao=aporte_padrao
)

historico_mensal_df, resumo = calcular_plano_evolucao(
    patrimonio_inicial=patrimonio_inicial,
    fluxo_df=fluxo_df,
    meta_etfs=meta_etfs,
    meses=meses
)

matriz_cov_anual = base_carteira.cov() * 12

pesos_finais = np.array([
    meta_renda_fixa / 100,
    meta_etfs / 100
])

oscilacao_final = np.sqrt(
    np.dot(
        pesos_finais.T,
        np.dot(matriz_cov_anual.values, pesos_finais)
    )
)

retorno_final = (
    pesos_finais[0] * ret_renda_fixa_aa +
    pesos_finais[1] * ret_etfs_aa
)

eficiencia_risco_retorno = retorno_final / oscilacao_final if oscilacao_final != 0 else np.nan

fronteira_df = calcular_fronteira_eficiente(
    retorno_renda_fixa=ret_renda_fixa_aa,
    retorno_etfs=ret_etfs_aa,
    matriz_cov_anual=matriz_cov_anual,
    pontos=101
)

melhor_eficiencia = fronteira_df.loc[fronteira_df["Eficiência risco-retorno"].idxmax()]


# =========================
# EVOLUÇÃO DA OSCILAÇÃO PELO PLANO
# =========================

historico_mensal_df["Oscilação Projetada"] = np.nan

for idx, linha in historico_mensal_df.iterrows():

    peso_renda_fixa = linha["% Renda Fixa"]
    peso_etfs = linha["% ETFs"]

    pesos_dinamicos = np.array([peso_renda_fixa, peso_etfs])

    oscilacao_dinamica = np.sqrt(
        np.dot(
            pesos_dinamicos.T,
            np.dot(matriz_cov_anual.values, pesos_dinamicos)
        )
    )

    historico_mensal_df.loc[idx, "Oscilação Projetada"] = oscilacao_dinamica


# =========================
# CENÁRIOS
# =========================

cenario_conservador = simular_cenario(
    historico_mensal_df,
    retorno_renda_fixa_aa=max(ret_renda_fixa_aa - 0.02, -0.10),
    retorno_etfs_aa=max(ret_etfs_aa - 0.06, -0.20)
)

cenario_base = simular_cenario(
    historico_mensal_df,
    retorno_renda_fixa_aa=ret_renda_fixa_aa,
    retorno_etfs_aa=ret_etfs_aa
)

cenario_otimista = simular_cenario(
    historico_mensal_df,
    retorno_renda_fixa_aa=ret_renda_fixa_aa + 0.02,
    retorno_etfs_aa=ret_etfs_aa + 0.06
)

cenarios_df = pd.DataFrame({
    "Cenário": ["Conservador", "Base", "Otimista"],
    "Retorno Renda Fixa a.a.": [max(ret_renda_fixa_aa - 0.02, -0.10), ret_renda_fixa_aa, ret_renda_fixa_aa + 0.02],
    "Retorno ETFs a.a.": [max(ret_etfs_aa - 0.06, -0.20), ret_etfs_aa, ret_etfs_aa + 0.06],
    "Patrimônio Final": [
        cenario_conservador.iloc[-1]["Patrimônio"],
        cenario_base.iloc[-1]["Patrimônio"],
        cenario_otimista.iloc[-1]["Patrimônio"]
    ],
    "% Final em ETFs": [
        cenario_conservador.iloc[-1]["% ETFs"],
        cenario_base.iloc[-1]["% ETFs"],
        cenario_otimista.iloc[-1]["% ETFs"]
    ]
})


# =========================
# SIMULAÇÃO PROBABILÍSTICA
# =========================

percentis_mc, resultados_mc = rodar_simulacao_probabilistica(
    historico_df=historico_mensal_df,
    media_mensal=media_mensal,
    cov_mensal=cov_mensal,
    n_simulacoes=n_simulacoes
)


# =========================
# SIMULAÇÃO DE CRISE
# =========================

cenarios_crise = {
    "Crise Global": {
        "Renda Fixa em USD": 0.04,
        "ETFs em USD": -0.25,
        "Variação do Dólar": 0.15
    },
    "Brasil Risk-Off": {
        "Renda Fixa em USD": 0.02,
        "ETFs em USD": -0.08,
        "Variação do Dólar": 0.20
    },
    "Abertura de Juros nos EUA": {
        "Renda Fixa em USD": -0.07,
        "ETFs em USD": -0.10,
        "Variação do Dólar": 0.08
    },
    "Cenário Positivo Global": {
        "Renda Fixa em USD": 0.04,
        "ETFs em USD": 0.12,
        "Variação do Dólar": -0.05
    }
}

crise_df = calcular_simulacao_crise(
    peso_renda_fixa=meta_renda_fixa / 100,
    peso_etfs=meta_etfs / 100,
    cenarios=cenarios_crise
)


# =========================
# ABAS
# =========================

aba_resumo, aba_ativos, aba_simulacoes = st.tabs(
    [
        "Resumo e Plano",
        "Ativos e Risco",
        "Cenários e Simulações"
    ]
)


# =========================
# ABA 1 — RESUMO E PLANO
# =========================

with aba_resumo:

    st.header("Resumo Executivo e Plano de Aportes")

    st.markdown(
        """
        Esta seção mostra como a carteira pode evoluir até a alocação desejada,
        indicando quanto deve ser direcionado mensalmente para Renda Fixa e para ETFs.

        A ideia é construir a exposição de forma gradual, sem necessidade de uma mudança brusca na carteira atual.
        """
    )

    pdf_relatorio = gerar_pdf_relatorio(
        patrimonio_inicial=patrimonio_inicial,
        data_final=data_final,
        meses=meses,
        meta_etfs=meta_etfs,
        meta_renda_fixa=meta_renda_fixa,
        resumo=resumo,
        oscilacao_final=oscilacao_final,
        eficiencia_risco_retorno=eficiencia_risco_retorno,
        referencia_renda_fixa=referencia_renda_fixa,
        etfs_validos=etfs_validos,
        cenarios_df=cenarios_df,
        percentis_mc=percentis_mc,
        crise_df=crise_df
    )

    if pdf_relatorio is not None:
        st.download_button(
            label="Baixar relatório em PDF",
            data=pdf_relatorio,
            file_name="relatorio_simulador_patrimonial_mwealth.pdf",
            mime="application/pdf",
            use_container_width=False
        )
    else:
        st.warning("Para habilitar o PDF, adicione reportlab no requirements.txt.")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Patrimônio inicial", formatar_moeda(patrimonio_inicial))

    with col2:
        st.metric("Patrimônio estimado sem rentabilidade", formatar_moeda(resumo["patrimonio_final_sem_rentabilidade"]))

    with col3:
        st.metric("Meta financeira em ETFs", formatar_moeda(resumo["meta_etfs_financeira"]))

    with col4:
        st.metric("Meses até a meta", meses)

    col5, col6, col7, col8 = st.columns(4)

    with col5:
        st.metric("Aporte médio em ETFs", formatar_moeda(resumo["aporte_mensal_medio_etfs"]))

    with col6:
        st.metric("Aporte médio em Renda Fixa", formatar_moeda(resumo["aporte_mensal_medio_renda_fixa"]))

    with col7:
        st.metric("Oscilação esperada da carteira", f"{oscilacao_final:.2%}")

    with col8:
        st.metric("Eficiência risco-retorno", f"{eficiencia_risco_retorno:.2f}")

    col_grafico_meta, col_grafico_oscilacao = st.columns(2)

    with col_grafico_meta:

        st.subheader("Evolução até a meta")

        fig_plano = go.Figure()

        fig_plano.add_trace(
            go.Scatter(
                x=historico_mensal_df["Data"],
                y=historico_mensal_df["% ETFs"] * 100,
                mode="lines+markers",
                name="% em ETFs"
            )
        )

        fig_plano.add_hline(
            y=meta_etfs,
            line_dash="dash",
            annotation_text="Meta final",
            annotation_position="top left"
        )

        fig_plano.update_layout(
            title="Exposição em ETFs",
            xaxis_title="Data",
            yaxis_title="% em ETFs",
            height=430
        )

        st.plotly_chart(fig_plano, use_container_width=True)

    with col_grafico_oscilacao:

        st.subheader("Evolução da oscilação esperada")

        fig_oscilacao = go.Figure()

        fig_oscilacao.add_trace(
            go.Scatter(
                x=historico_mensal_df["Data"],
                y=historico_mensal_df["Oscilação Projetada"] * 100,
                mode="lines+markers",
                name="Oscilação projetada"
            )
        )

        fig_oscilacao.update_layout(
            title="Oscilação anualizada estimada",
            xaxis_title="Data",
            yaxis_title="Oscilação anualizada (%)",
            height=430
        )

        st.plotly_chart(fig_oscilacao, use_container_width=True)

    st.markdown(
        """
        A oscilação projetada mostra como o risco da carteira tende a mudar conforme a exposição em ETFs aumenta ao longo do tempo.
        """
    )

    st.subheader("Plano mensal de aportes")

    historico_tabela = formatar_datas_tabela(historico_mensal_df, "Data")

    st.dataframe(
        historico_tabela.style.format({
            "Aporte Total": "R$ {:,.2f}",
            "Aporte em ETFs": "R$ {:,.2f}",
            "Aporte em Renda Fixa": "R$ {:,.2f}",
            "Saldo em ETFs": "R$ {:,.2f}",
            "Saldo em Renda Fixa": "R$ {:,.2f}",
            "Patrimônio Total": "R$ {:,.2f}",
            "% ETFs": "{:.2%}",
            "% Renda Fixa": "{:.2%}",
            "Oscilação Projetada": "{:.2%}"
        }),
        use_container_width=True
    )

    st.subheader("Distribuição mensal dos aportes")

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
            y=historico_mensal_df["Aporte em Renda Fixa"],
            name="Aporte em Renda Fixa"
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

    st.header("Ativos Utilizados e Comportamento Histórico")

    st.markdown(
        """
        Esta seção mostra os ativos usados como referência no estudo e o comportamento histórico deles.

        A Renda Fixa Internacional é representada por um ETF de Treasury americano líquido,
        usado apenas como referência de risco e retorno.
        """
    )

    etfs_texto = ", ".join(etfs_validos)

    col_a1, col_a2, col_a3 = st.columns(3)

    with col_a1:
        st.metric("Referência de Renda Fixa", referencia_renda_fixa)

    with col_a2:
        st.metric("ETFs utilizados", etfs_texto)

    with col_a3:
        st.metric("Relação entre Renda Fixa e ETFs", f"{relacao_renda_fixa_etfs:.2f}")

    metricas_risco = pd.DataFrame({
        "Classe": ["Renda Fixa Internacional em BRL", "ETFs Internacionais em BRL"],
        "Retorno Anual": [ret_renda_fixa_aa, ret_etfs_aa],
        "Oscilação Anual": [vol_renda_fixa_aa, vol_etfs_aa],
        "Maior queda histórica": [queda_renda_fixa, queda_etfs],
        "Perda mensal estimada em cenário ruim": [perda_ruim_renda_fixa, perda_ruim_etfs]
    })

    st.dataframe(
        metricas_risco.style.format({
            "Retorno Anual": "{:.2%}",
            "Oscilação Anual": "{:.2%}",
            "Maior queda histórica": "{:.2%}",
            "Perda mensal estimada em cenário ruim": "{:.2%}"
        }),
        use_container_width=True
    )

    col_matriz, col_perf = st.columns(2)

    with col_matriz:

        st.subheader("Matriz de risco entre as classes")

        matriz_risco_visual = matriz_cov_anual.copy()

        fig_matriz_risco = px.imshow(
            matriz_risco_visual,
            text_auto=".4f",
            aspect="auto",
            title="Matriz de covariância anualizada"
        )

        fig_matriz_risco.update_layout(height=430)

        st.plotly_chart(fig_matriz_risco, use_container_width=True)

    with col_perf:

        st.subheader("Performance histórica em reais")

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
            title="Retorno acumulado em BRL",
            yaxis_tickformat=".0%",
            xaxis_title="Data",
            yaxis_title="Retorno acumulado",
            height=430
        )

        st.plotly_chart(fig_perf, use_container_width=True)

    st.subheader("Fronteira eficiente")

    st.markdown(
        """
        A fronteira eficiente mostra combinações possíveis entre Renda Fixa Internacional e ETFs.
        A ideia é visualizar quanto retorno esperado a carteira pode buscar para diferentes níveis de oscilação.
        O ponto destacado representa a meta escolhida para esta simulação.
        """
    )

    fig_fronteira = px.scatter(
        fronteira_df,
        x="Oscilação esperada",
        y="Retorno esperado",
        color="% em ETFs",
        title="Combinações possíveis entre Renda Fixa e ETFs",
        labels={
            "Oscilação esperada": "Oscilação esperada",
            "Retorno esperado": "Retorno esperado",
            "% em ETFs": "% em ETFs"
        }
    )

    ponto_meta = pd.DataFrame({
        "Oscilação esperada": [oscilacao_final],
        "Retorno esperado": [retorno_final],
        "Carteira": ["Meta escolhida"]
    })

    fig_fronteira.add_trace(
        go.Scatter(
            x=ponto_meta["Oscilação esperada"],
            y=ponto_meta["Retorno esperado"],
            mode="markers+text",
            name="Meta escolhida",
            text=["Meta"],
            textposition="top center",
            marker=dict(size=14, symbol="star", color="#FFFFFF", line=dict(width=2, color="#7895E8"))
        )
    )

    fig_fronteira.add_trace(
        go.Scatter(
            x=[melhor_eficiencia["Oscilação esperada"]],
            y=[melhor_eficiencia["Retorno esperado"]],
            mode="markers+text",
            name="Maior eficiência histórica",
            text=["Mais eficiente"],
            textposition="bottom center",
            marker=dict(size=12, symbol="diamond", color="#7895E8")
        )
    )

    fig_fronteira.update_layout(
        xaxis_tickformat=".1%",
        yaxis_tickformat=".1%",
        height=460
    )

    st.plotly_chart(fig_fronteira, use_container_width=True)

    col_fronteira_1, col_fronteira_2, col_fronteira_3 = st.columns(3)

    with col_fronteira_1:
        st.metric("Meta escolhida", f"{meta_renda_fixa}% Renda Fixa / {meta_etfs}% ETFs")

    with col_fronteira_2:
        st.metric("Retorno esperado da meta", f"{retorno_final:.2%}")

    with col_fronteira_3:
        st.metric("Maior eficiência histórica", f"{melhor_eficiencia['% em ETFs']:.0%} em ETFs")

    with st.expander("Ver detalhes técnicos do modelo"):
        st.markdown(
            """
            Esta área concentra as bases quantitativas usadas nos cálculos.
            Ela serve para validação interna da equipe, não necessariamente para apresentação ao cliente.
            """
        )

        st.subheader("Base mensal de retornos")

        base_carteira_tabela = base_carteira.tail(24).copy()
        base_carteira_tabela.index = pd.to_datetime(base_carteira_tabela.index).strftime("%d/%m/%Y")

        st.dataframe(
            base_carteira_tabela.style.format("{:.2%}"),
            use_container_width=True
        )

        st.subheader("Matriz de relação entre os ativos")

        st.dataframe(
            base_carteira.cov().style.format("{:.6f}"),
            use_container_width=True
        )

        st.subheader("Matriz anualizada de risco")

        st.dataframe(
            matriz_cov_anual.style.format("{:.6f}"),
            use_container_width=True
        )

        st.subheader("Pontos da fronteira eficiente")

        fronteira_tabela = fronteira_df.copy()

        st.dataframe(
            fronteira_tabela.style.format({
                "% em ETFs": "{:.0%}",
                "% em Renda Fixa": "{:.0%}",
                "Retorno esperado": "{:.2%}",
                "Oscilação esperada": "{:.2%}",
                "Eficiência risco-retorno": "{:.2f}"
            }),
            use_container_width=True
        )

        st.subheader("Retornos mensais dos ETFs")

        retornos_etfs_tabela = retornos_etfs_usd.tail(24).copy()
        retornos_etfs_tabela.index = pd.to_datetime(retornos_etfs_tabela.index).strftime("%d/%m/%Y")

        st.dataframe(
            retornos_etfs_tabela.style.format("{:.2%}"),
            use_container_width=True
        )


# =========================
# ABA 3 — CENÁRIOS E SIMULAÇÕES
# =========================

with aba_simulacoes:

    st.header("Cenários e Simulações")

    st.markdown(
        """
        Esta seção concentra as análises prospectivas da carteira.  
        O objetivo é visualizar diferentes possibilidades de evolução patrimonial,
        desde cenários simples até simulações probabilísticas e cenários de crise.
        """
    )

    st.subheader("1. Cenários patrimoniais")

    st.markdown(
        """
        Os cenários patrimoniais usam hipóteses anuais de retorno para Renda Fixa e ETFs.
        Eles ajudam a visualizar uma trajetória conservadora, uma trajetória base
        e uma trajetória otimista até a data final do estudo.
        """
    )

    st.dataframe(
        cenarios_df.style.format({
            "Retorno Renda Fixa a.a.": "{:.2%}",
            "Retorno ETFs a.a.": "{:.2%}",
            "Patrimônio Final": "R$ {:,.2f}",
            "% Final em ETFs": "{:.2%}"
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

    st.subheader("2. Simulação probabilística")

    st.markdown(
        """
        Esta simulação gera milhares de caminhos possíveis para o patrimônio,
        com base no comportamento histórico da Renda Fixa Internacional e dos ETFs.

        O objetivo não é prever exatamente o futuro, mas mostrar uma faixa provável de resultados.
        """
    )

    col_mc1, col_mc2, col_mc3 = st.columns(3)

    with col_mc1:
        st.metric("Quantidade de simulações", n_simulacoes)

    with col_mc2:
        st.metric("Cenário central", formatar_moeda(percentis_mc.iloc[-1]["Cenário central"]))

    with col_mc3:
        st.metric("Cenário pessimista", formatar_moeda(percentis_mc.iloc[-1]["Cenário pessimista"]))

    fig_mc = go.Figure()

    for coluna in ["Cenário otimista", "Faixa superior", "Cenário central", "Faixa inferior", "Cenário pessimista"]:
        fig_mc.add_trace(
            go.Scatter(
                x=percentis_mc.index,
                y=percentis_mc[coluna],
                mode="lines",
                name=coluna
            )
        )

    fig_mc.update_layout(
        title="Faixa Provável de Evolução Patrimonial",
        xaxis_title="Data",
        yaxis_title="Patrimônio"
    )

    st.plotly_chart(fig_mc, use_container_width=True)

    with st.expander("Ver tabela da simulação probabilística"):
        percentis_tabela = percentis_mc.tail(12).copy()
        percentis_tabela.index = pd.to_datetime(percentis_tabela.index).strftime("%d/%m/%Y")

        st.dataframe(
            percentis_tabela.style.format("R$ {:,.2f}"),
            use_container_width=True
        )

    st.divider()

    st.subheader("3. Simulação de crise")

    st.markdown(
        """
        A simulação de crise mostra como a carteira poderia se comportar em cenários adversos,
        como queda dos mercados internacionais, alta do dólar ou abertura de juros nos Estados Unidos.
        """
    )

    st.dataframe(
        crise_df.style.format({
            "Renda Fixa em USD": "{:.2%}",
            "ETFs em USD": "{:.2%}",
            "Variação do Dólar": "{:.2%}",
            "Renda Fixa em BRL": "{:.2%}",
            "ETFs em BRL": "{:.2%}",
            "Impacto na Carteira": "{:.2%}"
        }),
        use_container_width=True
    )

    fig_crise = px.bar(
        crise_df,
        x="Cenário",
        y="Impacto na Carteira",
        title="Impacto Estimado em Cenários de Crise",
        text_auto=".2%"
    )

    fig_crise.update_layout(
        yaxis_tickformat=".0%"
    )

    st.plotly_chart(fig_crise, use_container_width=True)