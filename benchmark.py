import pandas as pd
import numpy as np
import time
import os
import requests  # Necessário para baixar os dados do benchmark
import seaborn as sns
import matplotlib.pyplot as plt
from amplpy import AMPL, Environment


def carregar_dados_or_library(nome_benchmark: str, periodos_por_ano: int):
    """
    Baixa e processa um arquivo de benchmark da OR-Library.

    Args:
        nome_benchmark (str): O nome do arquivo (ex: 'port1')
        periodos_por_ano (int): O número de períodos nos dados para anualizar 
                                (ex: 52 para dados semanais).

    Returns:
        (pd.Series, pd.DataFrame): Tupla contendo (mu_anual, Sigma_anual)
                                   Retorna (None, None) em caso de falha.
    """
    url = f"http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/{nome_benchmark}.txt"

    try:
        print(f"\nBaixando dados do benchmark: {url}")
        response = requests.get(url)
        response.raise_for_status()  # Verifica se o download foi OK

        linhas = response.text.strip().split('\n')

        print("Processando arquivo...")

        # 1. Ler o número de ativos (primeira linha)
        n_ativos = int(linhas[0].strip())

        # 2. Listas para guardar os dados
        retornos_periodo = []
        desvios_periodo = []
        # Tickers fictícios, já que os arquivos não têm nomes
        tickers = [f"Ativo_{i+1:03d}" for i in range(n_ativos)]

        # Inicializa a matriz de correlação como identidade
        matriz_correlacao = np.identity(n_ativos)

        # 3. Ler dados dos ativos (Retorno, Desvio)
        # As próximas N linhas são mu e sigma
        indice_linha_atual = 1
        for i in range(n_ativos):
            partes = linhas[indice_linha_atual + i].strip().split()
            retornos_periodo.append(float(partes[0]))
            desvios_periodo.append(float(partes[1]))

        indice_linha_atual += n_ativos

        # 4. Ler dados de correlação (o resto das linhas)
        for i in range(indice_linha_atual, len(linhas)):
            partes = linhas[i].strip().split()
            if len(partes) != 3:
                continue  # Ignora linhas mal formatadas

            idx_i = int(partes[0]) - 1  # Arquivos são 1-indexados
            idx_j = int(partes[1]) - 1  # Arquivos são 1-indexados
            correl = float(partes[2])

            matriz_correlacao[idx_i, idx_j] = correl
            matriz_correlacao[idx_j, idx_i] = correl  # Simetria

        # 5. Converter para DataFrames/Series do Pandas
        mu = pd.Series(retornos_periodo, index=tickers)
        sigma = pd.Series(desvios_periodo, index=tickers)
        df_correlacao = pd.DataFrame(
            matriz_correlacao, index=tickers, columns=tickers)

        # 6. Calcular a Matriz de Covariância
        # Formula: Sigma(i, j) = Correl(i, j) * Sigma(i) * Sigma(j)
        df_covariancia = df_correlacao.multiply(
            sigma, axis=0).multiply(sigma, axis=1)

        # 7. Anualizar os dados
        mu_anual = mu * periodos_por_ano
        Sigma_anual = df_covariancia * periodos_por_ano

        print(
            f"Dados do benchmark '{nome_benchmark}' carregados e processados com sucesso ({n_ativos} ativos).")
        return mu_anual, Sigma_anual

    except Exception as e:
        print(f"ERRO ao carregar dados do benchmark '{nome_benchmark}': {e}")
        return None, None


MODELO_AMPL_BENCHMARK_STRING = """
# --- 1. CONJUNTOS ---
set ATIVOS;

# --- 2. PARÂMETROS ---
param m; 
param W_max;
param W_min;
param R_target;
param mu {ATIVOS};
param Sigma {ATIVOS, ATIVOS};

# --- 3. VARIÁVEIS DE DECISÃO ---
var w {ATIVOS} >= 0; 
var b {ATIVOS} binary;

# --- 4. FUNÇÃO OBJETIVO ---
minimize Risco_Portfolio:
    sum {i in ATIVOS, j in ATIVOS} w[i] * Sigma[i,j] * w[j];

# --- 5. RESTRIÇÕES ---
subject to
    Soma_Pesos: sum {i in ATIVOS} w[i] = 1;
    Retorno_Alvo: sum {i in ATIVOS} mu[i] * w[i] >= R_target;
    Cardinalidade: sum {i in ATIVOS} b[i] = m;
    Aporte_Maximo {i in ATIVOS}: w[i] <= W_max * b[i];
    Aporte_Minimo {i in ATIVOS}: w[i] >= W_min * b[i];
;
"""


def resolver_com_ampl_benchmark(params, data, ampl_env):
    """Configura e resolve o modelo de benchmark (agora com W_min)."""
    mu, Sigma = data
    # Extrai os 4 parâmetros
    m, W_max, R_target, W_min = params['m'], params['W_max'], params['R_target'], params['W_min']

    try:
        ampl = AMPL(ampl_env)
        ampl.reset()
        ampl.eval(MODELO_AMPL_BENCHMARK_STRING)

        print("\nConfigurando o modelo AMPL (Benchmark) com os parâmetros...")
        ativos = Sigma.columns.tolist()
        ampl.set['ATIVOS'] = ativos

        ampl.param['m'] = m
        ampl.param['W_max'] = W_max
        ampl.param['W_min'] = W_min  # <-- PASSA O NOVO PARÂMETRO
        ampl.param['R_target'] = R_target
        ampl.param['mu'].set_values(mu)
        ampl.param['Sigma'].set_values(Sigma)

        ampl.option['solver'] = 'gurobi'
        ampl.option['gurobi_options'] = 'outlev=1 mipgap=0.01 timelimit=120'

        print("Iniciando o solver Gurobi via AMPL...")
        start_time = time.time()
        ampl.solve()
        solve_time = time.time() - start_time

        return ampl, solve_time

    except Exception as e:
        print(f"Ocorreu um erro durante a execução do AMPL (Benchmark): {e}")
        return None, None


# def resolver_com_ampl_benchmark(params, data, ampl_env):
#     """Configura e resolve o modelo de benchmark (sem setores)."""
#     mu, Sigma = data  # Note que este 'data' só terá mu e Sigma
#     m, W_max, R_target = params['m'], params['W_max'], params['R_target']

#     try:
#         ampl = AMPL(ampl_env)
#         ampl.reset()
#         ampl.eval(MODELO_AMPL_BENCHMARK_STRING)  # <-- USA O NOVO MODELO

#         print("\nConfigurando o modelo AMPL (Benchmark) com os parâmetros...")
#         ativos = Sigma.columns.tolist()

#         ampl.set['ATIVOS'] = ativos

#         # Não há NENHUMA linha de código sobre setores aqui

#         ampl.param['m'] = m
#         ampl.param['W_max'] = W_max
#         ampl.param['R_target'] = R_target
#         ampl.param['mu'].set_values(mu)
#         ampl.param['Sigma'].set_values(Sigma)

#         ampl.option['solver'] = 'gurobi'
#         # 5 min limite
#         ampl.option['gurobi_options'] = 'outlev=1 mipgap=0.01 timelimit=300'

#         print("Iniciando o solver Gurobi via AMPL...")
#         start_time = time.time()
#         ampl.solve()
#         solve_time = time.time() - start_time

#         return ampl, solve_time

#     except Exception as e:
#         print(f"Ocorreu um erro durante a execução do AMPL (Benchmark): {e}")
#         return None, None


def executar_analise_benchmark(dados_carregados, ampl_env, m_fixo, W_min_fixo, range_R_target, range_W_max, taxa_livre_risco):
    """
    Executa o modelo de benchmark várias vezes.
    """
    print("\n" + "="*60)
    print(" INICIANDO ANÁLISE DE SENSIBILIDADE (BENCHMARK)")
    print("="*60)

    resultados_analise = []
    total_runs = len(range_W_max) * len(range_R_target)
    run_count = 0

    for w_max in range_W_max:
        for r_target in range_R_target:
            run_count += 1
            print(
                f"\n[Execução {run_count}/{total_runs}] Testando W_max = {w_max:.2%} | R_target = {r_target:.2%}")

            params = {
                'm': m_fixo,
                'W_max': w_max,
                'R_target': r_target,
                'W_min': W_min_fixo  # <-- INCLUI O NOVO PARÂMETRO
            }

            # Chama o resolver (que agora sabe lidar com 4 parâmetros)
            ampl, solve_time = resolver_com_ampl_benchmark(
                params, dados_carregados, ampl_env)
            resultado_simulacao = {
                'W_max': w_max, 'R_target': r_target, 'm': m_fixo
            }

            status_sucesso = ['solved', 'limit']

            if ampl and ampl.get_value('solve_result') in status_sucesso:
                risco_portfolio = np.sqrt(
                    ampl.get_objective('Risco_Portfolio').value())
                retorno_portfolio = ampl.get_value(
                    'sum {i in ATIVOS} mu[i] * w[i]')
                sharpe_ratio = (retorno_portfolio -
                                taxa_livre_risco) / risco_portfolio

                resultado_simulacao.update({
                    'Status': 'Solucionado', 'Risco': risco_portfolio,
                    'Retorno': retorno_portfolio, 'Sharpe': sharpe_ratio
                })
                print(
                    f"--> Solução encontrada! Risco: {risco_portfolio:.2%}, Retorno: {retorno_portfolio:.2%}, Sharpe: {sharpe_ratio:.2f}")
            else:
                status = ampl.get_value('solve_result') if ampl else 'Erro'
                resultado_simulacao.update({
                    'Status': status, 'Risco': None,
                    'Retorno': None, 'Sharpe': None
                })
                print(
                    f"--> Não foi encontrada uma solução viável. Status: {status}")

            resultados_analise.append(resultado_simulacao)

    print("\n" + "="*60)
    print(" ANÁLISE DE BENCHMARK CONCLUÍDA")
    print("="*60)

    return pd.DataFrame(resultados_analise)


def carregar_gabarito_or_library(nome_arquivo_gabarito: str, periodos_por_ano: int):
    """
    Baixa e processa um arquivo de "gabarito" da fronteira eficiente
    (ex: 'portef1' ou 'portc1') da OR-Library, JÁ ANUALIZADO.

    Args:
        nome_arquivo_gabarito (str): O nome exato do arquivo (ex: 'portef1' ou 'portc1')
        periodos_por_ano (int): O número de períodos nos dados (ex: 52)

    Returns:
        pd.DataFrame: Um DataFrame com 'Retorno' e 'Risco' (anuais) do gabarito.
    """
    # URL agora usa o nome exato do arquivo (adicionamos .txt)
    url = f"http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/{nome_arquivo_gabarito}.txt"

    try:
        print(f"\nBaixando dados do GABARITO: {url}")
        response = requests.get(url)
        response.raise_for_status()

        linhas = response.text.strip().split('\n')

        pontos_fronteira = []
        for linha in linhas:
            partes = linha.strip().split()
            if len(partes) != 2:
                continue

            retorno_medio_periodo = float(partes[0])
            variancia_periodo = float(partes[1])

            # Anualiza os dados
            retorno_anual = retorno_medio_periodo * periodos_por_ano
            variancia_anual = variancia_periodo * periodos_por_ano
            risco_anual = np.sqrt(variancia_anual)

            pontos_fronteira.append({
                'Retorno': retorno_anual,
                'Risco': risco_anual
            })

        print(
            f"Gabarito '{nome_arquivo_gabarito}.txt' carregado e ANUALIZADO com sucesso.")
        return pd.DataFrame(pontos_fronteira)

    except Exception as e:
        print(
            f"ERRO ao carregar dados do gabarito '{nome_arquivo_gabarito}.txt': {e}")
        return None

if __name__ == '__main__':
    print("Iniciando Validação de Modelo RESTRINGIDO (Série Completa OR-Library)...")

    # --- 1. SETUP INICIAL ---
    try:
        ampl_env = Environment()
    except Exception:
        try:
            # Lembre-se de verificar se este é o caminho CORRETO da sua instalação
            caminho_ampl = r"C:\AMPL"
            ampl_env = Environment(caminho_ampl)
        except Exception as e:
            print(f"ERRO CRÍTICO: Não foi possível encontrar os executáveis do AMPL.")
            exit()

    # --- 2. LISTA DE TAREFAS ---
    # O '5' é o benchmark mais pesado (225 ativos)
    benchmarks_a_testar = [1, 2, 3, 4, 5]

    # --- 3. PARÂMETROS GLOBAIS DA ANÁLISE RESTRINGIDA ---
    # Parâmetros dos gabaritos 'portc' (m=10, W_min=0.01, W_max=1.0)
    m_analise = 10
    W_min_analise = 0.01
    pesos_maximos = [1.0]
    taxa_livre_risco = 0.03  # Usado para Sharpe, mas não crítico para o gráfico

    # --- 4. LOOP PRINCIPAL DE EXECUÇÃO ---
    # Este loop irá executar todo o processo para cada benchmark

    for i in benchmarks_a_testar:
        # Define os nomes dos arquivos dinamicamente
        nome_benchmark = f'port{i}'
        nome_gabarito_livre = f'portef{i}'
        nome_gabarito_restrito = f'portc{i}'
        nome_arquivo_grafico = f'validacao_benchmark_port{i}.png'

        print("\n" + "="*70)
        print(f"  PROCESSANDO BENCHMARK: {nome_benchmark.upper()}")
        print("="*70)

        # --- 4a. CARREGAMENTO DOS DADOS (3 FONTES) ---
        mu, Sigma = carregar_dados_or_library(
            nome_benchmark=nome_benchmark, periodos_por_ano=52)
        if mu is None:
            print(f"Erro ao carregar {nome_benchmark}. Pulando.")
            continue  # Pula para o próximo benchmark
        dados = (mu, Sigma)

        df_gabarito_livre = carregar_gabarito_or_library(
            nome_arquivo_gabarito=nome_gabarito_livre, periodos_por_ano=52)
        if df_gabarito_livre is None:
            print(f"Erro ao carregar {nome_gabarito_livre}. Pulando.")
            continue

        df_gabarito_restrito = carregar_gabarito_or_library(
            nome_arquivo_gabarito=nome_gabarito_restrito, periodos_por_ano=52)
        if df_gabarito_restrito is None:
            print(f"Erro ao carregar {nome_gabarito_restrito}. Pulando.")
            continue

        # --- 4b. DEFINIÇÃO DOS TARGETS (Baseado no gabarito restrito) ---
        # Se o gabarito restrito estiver vazio (ex: portc2 não existe), pulamos
        if df_gabarito_restrito.empty:
            print(
                f"Gabarito {nome_gabarito_restrito} está vazio ou não foi encontrado. Pulando.")
            continue

        targets_de_retorno = df_gabarito_restrito['Retorno'].values

        # --- 4c. EXECUÇÃO DA ANÁLISE ---
        print(
            f"\nIniciando execução do seu modelo em modo RESTRINGIDO (m=10, W_min=0.01) para {nome_benchmark}...")
        df_resultados_seu_modelo = executar_analise_benchmark(
            dados_carregados=dados,
            ampl_env=ampl_env,
            m_fixo=m_analise,
            W_min_fixo=W_min_analise,
            range_R_target=targets_de_retorno,
            range_W_max=pesos_maximos,
            taxa_livre_risco=taxa_livre_risco
        )

        # --- 4d. EXIBIÇÃO E COMPARAÇÃO ---
        if not df_resultados_seu_modelo.empty:
            df_solucionados = df_resultados_seu_modelo[df_resultados_seu_modelo['Status'] == 'Solucionado'].copy(
            )

            if not df_solucionados.empty:
                # PLOTANDO A COMPARAÇÃO TRIPLA
                print(
                    f"\nGerando gráfico de validação: {nome_arquivo_grafico}...")
                plt.figure(figsize=(12, 8))

                # 1. Gabarito NÃO-RESTRINGIDO (Azul)
                sns.scatterplot(
                    data=df_gabarito_livre, x='Risco', y='Retorno',
                    color='blue', s=150, marker='o', label=f'Gabarito Livre ({nome_gabarito_livre})'
                )

                # 2. Gabarito RESTRINGIDO (Verde)
                sns.scatterplot(
                    data=df_gabarito_restrito, x='Risco', y='Retorno',
                    color='green', s=150, marker='P', label=f'Gabarito Restrito ({nome_gabarito_restrito}, m=10)'
                )

                # 3. Seu Modelo RESTRINGIDO (Vermelho)
                sns.scatterplot(
                    data=df_solucionados, x='Risco', y='Retorno',
                    color='red', s=50, marker='X', label='Seu Modelo (m=10, W_min=0.01)'
                )

                plt.title(
                    f'Validação do Modelo Restringido vs. Gabarito ({nome_benchmark})', fontsize=16)
                plt.xlabel('Risco Anual (Volatilidade)', fontsize=12)
                plt.ylabel('Retorno Anual Esperado', fontsize=12)
                plt.gca().yaxis.set_major_formatter(
                    plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
                plt.gca().xaxis.set_major_formatter(
                    plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.6)

                # --- SALVAR O GRÁFICO EM VEZ DE MOSTRAR ---
                plt.savefig(nome_arquivo_grafico)
                plt.close()  # Fecha a figura para liberar memória e evitar sobreposição
                print(f"Gráfico salvo com sucesso em {nome_arquivo_grafico}")
            else:
                print(
                    f"\nSeu modelo não encontrou soluções para {nome_benchmark}.")

    print("\n" + "="*70)
    print("TODOS OS TESTES DE BENCHMARK FORAM CONCLUÍDOS.")
    print("="*70)
