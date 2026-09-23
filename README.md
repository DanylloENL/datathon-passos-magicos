# Datathon Passos Mágicos — Risco de Defasagem

Projeto do Tech Challenge (POSTECH) com a base de desenvolvimento educacional da
**Associação Passos Mágicos** (PEDE 2022–2024).

O objetivo é identificar antecipadamente alunos com risco de defasagem, para que a
associação possa oferecer acompanhamento antes que o atraso aconteça.

## Estrutura do repositório

```
├── app.py                  # Aplicação Streamlit
├── requirements.txt        # Bibliotecas da aplicação
├── modelo/
│   ├── modelo_risco_defasagem.pkl   # Modelo treinado (Random Forest)
│   ├── features_modelo.pkl          # Lista das variáveis de entrada
│   └── feature_importance.csv       # Importância de cada indicador
├── notebooks/
│   └── Código_pós.ipynb    # Treinamento e avaliação do modelo (Google Colab)
├── dados/
│   └── BASE DE DADOS PEDE 2024 - DATATHON (2).xlsx_4598.xlsx
└── analise_indicadores/
    └── Analise de indicadores.pdf   # Análise dos indicadores (IAN, IDA, IEG, IAA, IPS, IPP, IPV)
```

## Modelo preditivo

- **Pergunta:** a partir dos indicadores de um aluno em 2023, ele estará defasado em 2024?
- **Dados:** 765 alunos presentes nas bases PEDE 2023 e PEDE 2024, cruzados pelo RA.
- **Target:** aluno em risco quando a Defasagem em 2024 é menor que zero (40% dos alunos).
- **Entradas:** Idade, Ano de ingresso, IAA, IEG, IPS, IPP, IDA, IPV, IAN, INDE 2023 e Defasagem 2023.
- **Algoritmos comparados:** Regressão Logística, Árvore de Decisão e Random Forest.
- **Modelo escolhido:** Random Forest (maior ROC-AUC).

| Métrica | Teste (153 alunos) | Validação cruzada (5 folds) |
|---|---|---|
| Acurácia | 85,0% | 80,3% |
| Precisão | 80,0% | 73,4% |
| Recall | 83,9% | 80,2% |
| F1-score | 81,9% | 76,6% |
| ROC-AUC | 91,8% | 88,2% |

Os indicadores mais importantes para o modelo foram **IPP** (24%), **INDE 2023** (14%),
**IPV** (10%) e **Defasagem 2023** (10%).

## Glossário

| Sigla | Significado |
|---|---|
| PEDE | Pesquisa Extensiva do Desenvolvimento Educacional |
| RA | Registro do Aluno |
| INDE | Índice do Desenvolvimento Educacional |
| IAN | Indicador de Adequação ao Nível |
| IDA | Indicador de Desempenho Acadêmico |
| IEG | Indicador de Engajamento |
| IAA | Indicador de Autoavaliação |
| IPS | Indicador Psicossocial |
| IPP | Indicador Psicopedagógico |
| IPV | Indicador de Ponto de Virada |

## Como rodar

### Notebook

Abra `notebooks/Código_pós.ipynb` no Google Colab, execute todas as células e, quando
solicitado, envie a planilha da pasta `dados/`.

### Aplicação Streamlit

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

A aplicação abre em `http://localhost:8501` e possui três abas:

- **Previsão individual:** informa os indicadores de um aluno e retorna a probabilidade de risco.
- **Previsão em lote:** recebe uma planilha com vários alunos e retorna todos ordenados por risco.
- **Sobre o modelo:** metodologia, glossário, métricas e importância dos indicadores.
