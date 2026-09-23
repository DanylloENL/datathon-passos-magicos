from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ================================================================
# DATATHON - PASSOS MÁGICOS
# APLICAÇÃO STREAMLIT - PREVISÃO DE RISCO DE DEFASAGEM
# ================================================================

st.set_page_config(
    page_title="Risco de Defasagem - Passos Mágicos",
    page_icon="🎓",
    layout="wide"
)


# ----------------------------------------------------------------
# 1. CARREGAMENTO DO MODELO
# ----------------------------------------------------------------

PASTA_MODELO = Path(__file__).parent / "modelo"


@st.cache_resource
def carregar_modelo():

    modelo = joblib.load(PASTA_MODELO / "modelo_risco_defasagem.pkl")

    features = joblib.load(PASTA_MODELO / "features_modelo.pkl")

    return modelo, features


@st.cache_data
def carregar_importancia():

    try:
        return pd.read_csv(
            PASTA_MODELO / "feature_importance.csv",
            encoding="utf-8-sig"
        )
    except FileNotFoundError:
        return None


modelo, features = carregar_modelo()

importancia = carregar_importancia()


# ----------------------------------------------------------------
# 2. DESCRIÇÃO DOS INDICADORES
# ----------------------------------------------------------------

descricoes = {
    "IAA": "Indicador de Autoavaliação",
    "IEG": "Indicador de Engajamento",
    "IPS": "Indicador Psicossocial",
    "IPP": "Indicador Psicopedagógico",
    "IDA": "Indicador de Desempenho Acadêmico",
    "IPV": "Indicador de Ponto de Virada",
    "IAN": "Indicador de Adequação ao Nível",
    "INDE 2023": "Índice do Desenvolvimento Educacional (2023)",
}


# ----------------------------------------------------------------
# 3. CABEÇALHO
# ----------------------------------------------------------------

st.title("🎓 Previsão de Risco de Defasagem")

st.markdown(
    """
    Modelo de Machine Learning desenvolvido para o **Datathon Passos Mágicos**.

    A partir dos indicadores de um aluno em **2023**, o modelo estima a
    probabilidade de ele apresentar **defasagem em 2024**.
    """
)

aba_individual, aba_lote, aba_modelo = st.tabs([
    "Previsão individual",
    "Previsão em lote",
    "Sobre o modelo"
])


# ----------------------------------------------------------------
# 4. PREVISÃO INDIVIDUAL
# ----------------------------------------------------------------

with aba_individual:

    st.subheader("Indicadores do aluno (2023)")

    with st.form("form_aluno"):

        col1, col2, col3 = st.columns(3)

        with col1:

            idade = st.number_input(
                "Idade",
                min_value=5,
                max_value=30,
                value=12,
                step=1
            )

            ano_ingresso = st.number_input(
                "Ano de ingresso",
                min_value=2010,
                max_value=2023,
                value=2021,
                step=1
            )

            defasagem = st.number_input(
                "Defasagem em 2023",
                min_value=-10,
                max_value=10,
                value=0,
                step=1,
                help="Negativo = aluno abaixo da fase ideal."
            )

            inde = st.number_input(
                "INDE 2023",
                min_value=0.0,
                max_value=10.0,
                value=7.0,
                step=0.1,
                help=descricoes["INDE 2023"]
            )

        valores_indicadores = {}

        indicadores = ["IAA", "IEG", "IPS", "IPP", "IDA", "IPV", "IAN"]

        for i, indicador in enumerate(indicadores):

            coluna = col2 if i < 4 else col3

            with coluna:

                valores_indicadores[indicador] = st.number_input(
                    indicador,
                    min_value=0.0,
                    max_value=10.0,
                    value=7.0,
                    step=0.1,
                    help=descricoes[indicador]
                )

        enviar = st.form_submit_button(
            "Calcular risco",
            type="primary"
        )

    if enviar:

        aluno = {
            "Idade": idade,
            "Ano ingresso": ano_ingresso,
            **valores_indicadores,
            "INDE 2023": inde,
            "Defasagem_2023": defasagem,
        }

        X_aluno = pd.DataFrame([aluno])[features]

        probabilidade = modelo.predict_proba(X_aluno)[0][1]

        previsao = modelo.predict(X_aluno)[0]

        st.divider()

        col_a, col_b = st.columns(2)

        with col_a:

            st.metric(
                "Probabilidade de risco em 2024",
                f"{probabilidade:.1%}"
            )

            st.progress(float(probabilidade))

        with col_b:

            if previsao == 1:
                st.error("⚠️ ALUNO EM RISCO DE DEFASAGEM")
            else:
                st.success("✅ ALUNO SEM RISCO DE DEFASAGEM")

        st.caption(
            "A previsão é um apoio à decisão pedagógica e não substitui "
            "a avaliação da equipe da Passos Mágicos."
        )


# ----------------------------------------------------------------
# 5. PREVISÃO EM LOTE
# ----------------------------------------------------------------

with aba_lote:

    st.subheader("Previsão para vários alunos")

    st.markdown(
        "Envie um arquivo **Excel** ou **CSV** com as colunas abaixo. "
        "Também é possível enviar a planilha original do Datathon: "
        "a aba **PEDE2023** será utilizada automaticamente."
    )

    st.code(", ".join(features))

    arquivo = st.file_uploader(
        "Arquivo com os alunos",
        type=["xlsx", "xls", "csv"]
    )

    if arquivo is not None:

        if arquivo.name.lower().endswith(".csv"):

            dados = pd.read_csv(arquivo)

        else:

            abas = pd.ExcelFile(arquivo).sheet_names

            aba = "PEDE2023" if "PEDE2023" in abas else abas[0]

            dados = pd.read_excel(arquivo, sheet_name=aba)

            st.info(f"Aba utilizada: {aba}")

        # Na planilha original a coluna se chama "Defasagem"
        if "Defasagem_2023" not in dados.columns and "Defasagem" in dados.columns:
            dados["Defasagem_2023"] = dados["Defasagem"]

        faltando = [f for f in features if f not in dados.columns]

        if faltando:

            st.error(
                "Colunas não encontradas no arquivo: " + ", ".join(faltando)
            )

        else:

            X_lote = dados[features].apply(
                pd.to_numeric,
                errors="coerce"
            )

            dados["Probabilidade de risco"] = modelo.predict_proba(X_lote)[:, 1]

            dados["Classificação"] = np.where(
                modelo.predict(X_lote) == 1,
                "Em risco",
                "Sem risco"
            )

            colunas_id = [c for c in ["RA", "Nome Anonimizado", "Nome"] if c in dados.columns]

            resultado = dados[
                colunas_id + features + ["Probabilidade de risco", "Classificação"]
            ].sort_values("Probabilidade de risco", ascending=False)

            total = len(resultado)

            em_risco = (resultado["Classificação"] == "Em risco").sum()

            c1, c2, c3 = st.columns(3)

            c1.metric("Alunos analisados", total)
            c2.metric("Em risco", em_risco)
            c3.metric("Percentual em risco", f"{em_risco / total:.1%}")

            st.dataframe(
                resultado.style.format({"Probabilidade de risco": "{:.1%}"}),
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "Baixar resultado (CSV)",
                resultado.to_csv(index=False, encoding="utf-8-sig"),
                file_name="previsao_risco_defasagem.csv",
                mime="text/csv"
            )


# ----------------------------------------------------------------
# 6. SOBRE O MODELO
# ----------------------------------------------------------------

with aba_modelo:

    st.subheader("Como o modelo funciona")

    st.markdown(
        """
        - **Algoritmo:** Random Forest (selecionado entre Regressão Logística,
          Árvore de Decisão e Random Forest pelo maior ROC-AUC).
        - **Dados:** alunos presentes nas bases PEDE 2023 e PEDE 2024,
          cruzados pelo RA.
        - **Target:** aluno em risco quando a Defasagem em 2024 é menor que zero.
        - **Entradas:** somente indicadores de 2023, para não usar
          informação do futuro na previsão.
        - **Valores ausentes:** preenchidos com a mediana do treinamento.
        """
    )

    st.subheader("Glossário")

    st.markdown(
        """
        | Sigla | Significado | O que mede |
        |---|---|---|
        | **PEDE** | Pesquisa Extensiva do Desenvolvimento Educacional | Avaliação anual feita pela Passos Mágicos com todos os alunos |
        | **RA** | Registro do Aluno | Número de identificação do aluno, usado para ligar 2023 a 2024 |
        | **INDE** | Índice do Desenvolvimento Educacional | Nota geral do aluno: média ponderada dos indicadores abaixo |
        | **IAN** | Indicador de Adequação ao Nível | Se o aluno está na fase certa para a idade (ligado à defasagem) |
        | **IDA** | Indicador de Desempenho Acadêmico | Notas nas provas (Português, Matemática, Inglês) |
        | **IEG** | Indicador de Engajamento | Entrega de tarefas e participação nas atividades |
        | **IAA** | Indicador de Autoavaliação | Como o próprio aluno avalia a si mesmo e seus estudos |
        | **IPS** | Indicador Psicossocial | Avaliação da equipe de psicologia sobre aspectos emocionais e sociais |
        | **IPP** | Indicador Psicopedagógico | Avaliação sobre o desenvolvimento da aprendizagem do aluno |
        | **IPV** | Indicador de Ponto de Virada | Se o aluno está no momento de "virada" pela educação |
        | **Defasagem** | Fase atual − fase ideal para a idade | Negativo = aluno atrasado |

        Os indicadores e o INDE vão de **0 a 10**.
        """
    )

    st.subheader("Desempenho do modelo")

    st.markdown(
        """
        Resultados no **conjunto de teste** (153 alunos que o modelo não viu no treino):
        """
    )

    m1, m2, m3, m4, m5 = st.columns(5)

    m1.metric("Acurácia", "85,0%")
    m2.metric("Precisão", "80,0%")
    m3.metric("Recall", "83,9%")
    m4.metric("F1-score", "81,9%")
    m5.metric("ROC-AUC", "91,8%")

    st.markdown(
        """
        | Métrica | O que significa |
        |---|---|
        | **Acurácia** | De todos os alunos, quantos o modelo acertou |
        | **Precisão** | Dos alunos que o modelo apontou como "em risco", quantos realmente estavam |
        | **Recall** | Dos alunos que realmente estavam em risco, quantos o modelo encontrou |
        | **F1-score** | Equilíbrio entre precisão e recall |
        | **ROC-AUC** | Capacidade de separar alunos em risco dos sem risco (50% = sorteio, 100% = perfeito) |

        Na **validação cruzada** (5 rodadas com divisões diferentes dos dados), o modelo
        manteve ROC-AUC médio de **88,2%** e recall médio de **80,2%**, o que mostra
        que o resultado é estável.
        """
    )

    if importancia is not None:

        st.subheader("Importância dos indicadores")

        st.bar_chart(
            importancia.set_index("Feature")["Importância (%)"],
            horizontal=True,
            sort="-Importância (%)"
        )

        st.caption(
            "A importância indica quais variáveis o modelo mais utilizou "
            "nas previsões. Ela não significa relação de causa."
        )
