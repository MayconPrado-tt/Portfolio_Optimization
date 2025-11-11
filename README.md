# Otimização de Portfólio (MIQP) com Restrições de Cardinalidade

> Um pipeline completo em Python para otimização de carteiras de investimento, desde a limpeza de dados brutos até a validação do modelo (MIQP) contra benchmarks da literatura usando `amplpy` e `Gurobi`.

Este projeto implementa um modelo avançado de otimização de portfólio que estende a Teoria Moderna de Markowitz para incluir restrições do mundo real, como **limite de cardinalidade (`m`)** e **peso máximo por ativo (`W_max`)**.

Isso transforma o problema clássico de Programa Quadrático (QP) em um **Programa Quadrático Inteiro-Misto (MIQP)**, que é resolvido usando a linguagem de modelagem AMPL e o solver Gurobi.

O repositório está dividido em dois fluxos de trabalho principais:

1.  **Análise de Mercado (Dados Reais):** Um pipeline para carregar seus próprios dados de preços, limpá-los, calcular parâmetros (`mu`, `Sigma`) e rodar a análise de Fronteira Eficiente com regras setoriais complexas.
2.  **Validação Acadêmica (Benchmark):** Um script auto-contido que valida a corretude do modelo matemático contra os dados clássicos da OR-Library (`port1`, `portef1`).

-----

## 🛠️ Ferramentas e Metodologia

  * **Linguagem:** Python 3.x
  * **Modelagem:** `amplpy` (Python API para AMPL)
  * **Solver:** Gurobi Optimizer (via AMPL)
  * **Análise de Dados:** `pandas` e `numpy`
  * **Visualização:** `matplotlib` e `seaborn`
  * **Aquisição de Dados:** `requests` (para benchmarks), `yfinance` (para setores)

-----

## 📂 Estrutura do Projeto e Fluxo de Trabalho

Recomenda-se organizar os arquivos da seguinte forma para maior clareza:

```
/Portfolio_Optimization
|
|--- /Scripts de Preparação
|    |--- Limpa dados.py
|    |--- gerar_classificacao.py
|
|--- /Scripts de Otimização
|    |--- menu_otimizacao_ampl.py
|    |--- benchmark.py
|
|--- /Dados de Entrada (Exemplos)
|    |--- precos_fechamento_3_anos.csv
|
|--- /Dados Processados (Saída)
|    |--- vetor_retornos_calculado.csv
|    |--- matriz_covariancia_calculada.csv
|    |--- mapeamento_setores.csv
|
|--- README.md
|--- requirements.txt
```

### Fluxo 1: Análise de Mercado (Dados Reais)

Este é o fluxo principal para analisar sua própria carteira (ex: S\&P 500).

**Passo 1:** Forneça seus preços brutos em `precos_fechamento_3_anos.csv`.

**Passo 2:** Execute os scripts de preparação:

  * `python "Scripts de Preparação/Limpa dados.py"`: Lê os preços brutos, calcula os retornos diários, anualiza `mu` e `Sigma`, e salva `vetor_retornos_calculado.csv` e `matriz_covariancia_calculada.csv`.
  * `python "Scripts de Preparação/gerar_classificacao.py"`: Lê uma lista de tickers (provavelmente do arquivo de preços) e busca seus setores de mercado (via `yfinance`), salvando em `mapeamento_setores.csv`.

**Passo 3:** Execute o script de otimização principal:

  * `python "Scripts de Otimização/menu_otimizacao_ampl.py"`: Este script consome os 3 arquivos gerados no Passo 2 e roda a análise completa da Fronteira Eficiente, com todas as regras setoriais.

### Fluxo 2: Validação Acadêmica (Benchmark)

Este fluxo é usado para provar que o modelo matemático central (a função objetivo) está correto.

**Passo Único:** Execute o script de benchmark:

  * `python "Scripts de Otimização/benchmark.py"`: Este script é **auto-contido**. Ele ignora todos os arquivos CSV locais, baixa os dados (`port1`) e o gabarito (`portef1`) da web, anualiza ambos e plota um gráfico comparando os resultados do seu modelo com o gabarito da literatura.

-----

## 📄 Descrição Detalhada dos Arquivos

### Scripts de Preparação

  * **`Limpa dados.py`**:

      * **Entrada:** `precos_fechamento_3_anos.csv`
      * **O que faz:** Calcula os retornos diários (`.pct_change()`), anualiza o retorno médio (`.mean() * 252`) e a matriz de covariância (`.cov() * 252`).
      * **Saída:** `vetor_retornos_calculado.csv` (`mu`) e `matriz_covariancia_calculada.csv` (`Sigma`).

  * **`gerar_classificacao.py`**:

      * **Entrada:** (Provavelmente `precos_fechamento_3_anos.csv` para a lista de tickers).
      * **O que faz:** Itera sobre os tickers, usa a biblioteca `yfinance` para buscar o setor GICS de cada um.
      * **Saída:** `mapeamento_setores.csv`.

### Scripts de Otimização

  * **`menu_otimizacao_ampl.py`**:

      * **Entrada:** `vetor_retornos_calculado.csv`, `matriz_covariancia_calculada.csv`, `mapeamento_setores.csv`.
      * **O que faz:** Script principal para análise de mercado. Carrega os dados, aplica o modelo MIQP completo (incluindo regras de setores) e executa a "Análise de Sensibilidade" (mapeamento da Fronteira Eficiente), variando `R_target` e `W_max`.
      * **Saída:** Gráficos da Fronteira Eficiente e tabelas no console com as carteiras de maior Índice de Sharpe.

  * **`benchmark.py`**:

      * **Entrada:** Nenhuma (baixa dados da web).
      * **O que faz:** Valida o modelo-base. Roda uma versão simplificada do modelo (sem setores) com restrições "desligadas" (`m = N`, `W_max = 1.0`) e compara o resultado com o gabarito acadêmico (`portef1`).
      * **Saída:** Um gráfico de validação. Se os pontos do seu modelo (vermelhos) se sobrepõem aos do gabarito (azuis), o modelo está correto.

### Arquivos de Dados

  * **`/Dados de Entrada`**:

      * `precos_fechamento_3_anos.csv`: **(Necessário fornecer)** Arquivo CSV com preços de fechamento diários. As colunas devem ser os tickers.

  * **`/Dados Processados`**:

      * `vetor_retornos_calculado.csv`: Saída do `Limpa dados.py`.
      * `matriz_covariancia_calculada.csv`: Saída do `Limpa dados.py`.
      * `mapeamento_setores.csv`: Saída do `gerar_classificacao.py`.

-----

## 🏁 Como Rodar o Projeto (Guia Rápido)

### 1\. Requisitos

**Software Essencial:**

  * Python 3.8+
  * Uma instalação funcional do **AMPL**
  * Um solver de otimização, como **Gurobi** (com uma licença válida)

**Bibliotecas Python:**
(Salve isso como `requirements.txt`)

```
pandas
numpy
amplpy
matplotlib
seaborn
requests
lxml
yfinance
```

Instale com:

```bash
pip install -r requirements.txt
```

### 2\. Configuração do AMPL/Gurobi

Em **ambos** os scripts (`benchmark.py` e `menu_otimizacao_ampl.py`), encontre o bloco `if __name__ == '__main__':` e ajuste o `caminho_ampl` para apontar para a **pasta** onde seu `ampl.exe` está instalado.

```python
    try:
        ampl_env = Environment() # Tenta o PATH
    except Exception:
        try:
            # --- EDITE ESTA LINHA ---
            caminho_ampl = r"C:\ampl_mswin64" # Coloque o caminho da SUA instalação
            ampl_env = Environment(caminho_ampl)
        except Exception as e:
            print(f"ERRO CRÍTICO: Não foi possível encontrar os executáveis do AMPL.")
            exit()
```

### 3\. Passo a Passo da Execução

#### Passo A: Validar o Modelo (Recomendado)

Primeiro, confirme que seu ambiente e seu modelo-base estão corretos.

```bash
python "Scripts de Otimização/benchmark.py"
```

  * **O que esperar:** Um gráfico comparando seu modelo com o gabarito. Os pontos vermelhos e azuis devem se sobrepor perfeitamente.

#### Passo B: Preparar Dados de Mercado

Forneça seu arquivo `precos_fechamento_3_anos.csv` na pasta de entrada.

1.  **Gere `mu` e `Sigma`:**
    ```bash
    python "Scripts de Preparação/Limpa dados.py"
    ```
2.  **Gere o mapeamento de setores:**
    ```bash
    python "Scripts de Preparação/gerar_classificacao.py"
    ```

#### Passo C: Executar a Análise de Mercado

Com os três arquivos (`vetor...`, `matriz...`, `mapeamento...`) prontos na pasta de saída, rode a análise principal.

```bash
python "Scripts de Otimização/menu_otimizacao_ampl.py"
```

  * **O que esperar:** O script iniciará a Análise de Sensibilidade, mostrando o log do Gurobi para cada execução e, ao final, exibirá o gráfico da Fronteira Eficiente para seus dados.

(Fim do `README.md`)
