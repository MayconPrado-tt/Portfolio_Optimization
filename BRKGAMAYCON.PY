import numpy as np
import pandas as pd
import random
from typing import Dict, List, Tuple

# ============================================================
# 1. CARREGAR DADOS (mu, Sigma, setores)
# ============================================================

# Arquivos gerados por você na etapa de limpeza
ARQ_MU = "vetor_retornos_calculado.csv"
ARQ_SIGMA = "matriz_covariancia_calculada.csv"
ARQ_SETORES = "mapeamento_setores.csv"

# === Carregar vetor de retornos anuais ===
mu = pd.read_csv(ARQ_MU, index_col=0)

# Se vier como DataFrame com uma coluna só → vira Series
if isinstance(mu, pd.DataFrame) and mu.shape[1] == 1:
    mu = mu.iloc[:, 0]

# === Carregar matriz de covariância anual ===
Sigma = pd.read_csv(ARQ_SIGMA, index_col=0)

# Reordenar Sigma para bater com a ordem dos tickers em mu
tickers = list(mu.index)
Sigma = Sigma.loc[tickers, tickers]

# Lê mapeamento de setores
map_setores = pd.read_csv(ARQ_SETORES)

# Garante que só usamos tickers que estão em mu/Sigma
map_setores = map_setores[map_setores["Ticker"].isin(tickers)]

# Dicionário Ticker -> Setor bruto
ticker_to_sector: Dict[str, str] = dict(
    zip(map_setores["Ticker"], map_setores["Setor"])
)

# Para tickers sem setor mapeado (caso raro), marca como 'Indefinido'
for t in tickers:
    if t not in ticker_to_sector:
        ticker_to_sector[t] = "Indefinido"

n = len(tickers)
print(f"Número de ativos carregados: {n}")

# ============================================================
# 2. PARÂMETROS DO MODELO
# ============================================================

m = 15           # cardinalidade 
W_MAX = 0.10     # peso máximo por ativo (K ativos, ta simplificando o problema) nao definir pesos 0,9 0,5 0,5)
R_TARGET = 0.15  # retorno mínimo desejado (15% a.a.)

# ============================================================
# 3. DEFINIÇÃO DOS GRUPOS SETORIAIS (CORRIGIDO = IGUAL AO AMPL)
# ============================================================

# Função para normalizar nomes
def normalizar_setor(nome: str) -> str:
    if pd.isna(nome):
        return "Indefinido"
    nome = nome.strip()

    # igual ao AMPL
    if nome == "Consumer Defensive":
        return "Consumer Staples"  # Yahoo chama "Consumer Defensive"

    if nome in ["Healthcare", "Health Care"]:
        return "Health Care"

    if nome in ["Basic Materials", "Materials"]:
        return "Materials"

    if nome in ["Technology", "Information Technology"]:
        return "Information Technology"

    if nome in ["Financial Services", "Financials"]:
        return "Financials"

    return nome

# Normaliza setores
ticker_to_sector = {
    t: normalizar_setor(s) for t, s in ticker_to_sector.items()
}

# Conjunto total de setores presentes
setores_presentes = sorted(set(ticker_to_sector.values()))
print("Setores presentes nos dados:")
print(setores_presentes)

# === Grupos ===

SET_DEF = {
    "Consumer Staples",
    "Health Care",
    "Utilities",
}

SET_CIC = {
    "Consumer Discretionary",
    "Financials",
    "Industrials",
    "Real Estate",
    "Information Technology",
    "Communication Services",
}

SET_COM = {
    "Energy",
    "Materials",
}

SET_SENS = {
    "Financials",
    "Utilities",
    "Real Estate",
}

SET_TEC = {
    "Information Technology",
}

SET_ENE = {
    "Utilities",
}


# Conjunto total de setores S
S_ALL = sorted(setores_presentes)

# Map: setor -> lista de índices de ativos (i) que estão naquele setor
sector_to_assets: Dict[str, List[int]] = {s: [] for s in S_ALL}
for idx, t in enumerate(tickers):
    s = ticker_to_sector[t]
    sector_to_assets[s].append(idx)

# ============================================================
# 4. CÁLCULO DE MÉTRICAS DA CARTEIRA
# ============================================================

mu_vec = mu.values
Sigma_mat = Sigma.values

def carteira_metrics(w: np.ndarray,
                     ativos_selecionados: np.ndarray,
                     verbose: bool = False) -> Dict:
    """
    Calcula métricas da carteira:
      - retorno
      - risco (variância)
      - setores presentes (y_s)
      - contagens por grupo
    """
    # Retorno e risco
    retorno = float(np.dot(w, mu_vec))
    risco = float(w @ Sigma_mat @ w)

    # Descobre setores presentes
    y_s = {s: 0 for s in S_ALL} # esse setor nao aparece na carteira
    for idx in ativos_selecionados:
        s = ticker_to_sector[tickers[idx]] # descobrir o setor do ativo 
        y_s[s] = 1

    # Contagens por grupo, quero saber quais setores aparece no portifólio 
    n_def = sum(y_s.get(s, 0) for s in SET_DEF)
    n_cic = sum(y_s.get(s, 0) for s in SET_CIC)
    n_com = sum(y_s.get(s, 0) for s in SET_COM)
    n_sens = sum(y_s.get(s, 0) for s in SET_SENS)
    n_tec = sum(y_s.get(s, 0) for s in SET_TEC)
    n_ene = sum(y_s.get(s, 0) for s in SET_ENE)
    n_total = sum(y_s.values())

    if verbose:
        print("Retorno:", retorno)
        print("Risco:", risco)
        print("Setores presentes:", [s for s, v in y_s.items() if v == 1])

    return {
        "retorno": retorno,
        "risco": risco,
        "y_s": y_s,
        "n_def": n_def,
        "n_cic": n_cic,
        "n_com": n_com,
        "n_sens": n_sens,
        "n_tec": n_tec,
        "n_ene": n_ene,
        "n_total": n_total,
    }

# ============================================================
# 5. DECODER (Random Keys -> (w, b, y))
# ============================================================

def decoder(random_keys: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Decoder do BRKGA:
      - Ordena ativos pela chave aleatória (maior -> mais preferido).
      - Seleciona os m primeiros (b_i = 1).
      - Define pesos iguais w_i = 1/m para os selecionados.
      - Calcula métricas e retorna.
    """
    # Ordena índices de ativos
    order = np.argsort(-random_keys)  # decrescente 500 representar so 15 primeiro ativio 
    selecionados = order[:m] # 15 primeiros 

    # Vetores w e b
    w = np.zeros(n)
    b = np.zeros(n, dtype=int) # quais ativos foram selecionados inteiros 

    for idx in selecionados:
        b[idx] = 1
        w[idx] = 1.0 / m

    # Garantia simples: soma=1
    w = w / w.sum()

    # Cálculo de métricas
    metrics = carteira_metrics(w, selecionados) #chamando a função da carteira 

    return w, b, metrics

# ============================================================
# 6. FUNÇÃO DE FITNESS COM PENALIZAÇÃO
# ============================================================

def fitness(random_keys: np.ndarray) -> float:
    """
    Fitness a ser minimizado:
      f = risco + penalidades
    Penalidades:
      - retorno < R_TARGET
      - violações de regras de diversificação setorial
    """
    w, b, met = decoder(random_keys) # metricas calculadas pelo decoder
    # exemplos met, met = { "retorno": 0.18,"risco": 0.045,"y_s": {"Energy":1, "Tech":1, "Health Care":1},"n_def": 1,"n_cic": 2,"n_com": 0,
  
    risco = met["risco"] # número da variancia 
    ret = met["retorno"] # retorno calculado 

    # Penalidade base
    pen = 0.0

    # (1) Retorno mínimo
    if ret < R_TARGET:
        pen += 1000.0 * (R_TARGET - ret)  # Se o retorno da carteira for MENOR que o mínimo desejado (R_TARGET = 15%)

    # (2) Regras de diversificação
    # Pelo menos um setor defensivo
    if met["n_def"] < 1:
        pen += 100.0 * (1 - met["n_def"])

    # Pelo menos um setor cíclico
    if met["n_cic"] < 1:
        pen += 100.0 * (1 - met["n_cic"])

    # No máximo um setor de commodities
    if met["n_com"] > 1:
        pen += 100.0 * (met["n_com"] - 1)

    # No máximo dois setores sensíveis a juros
    if met["n_sens"] > 2:
        pen += 100.0 * (met["n_sens"] - 2)

    # Tecnologia só se houver energia elétrica
    if met["n_tec"] >= 1 and met["n_ene"] == 0:
        pen += 200.0

    # Pelo menos 4 setores no total
    if met["n_total"] < 4:
        pen += 100.0 * (4 - met["n_total"])

    # Fitness final
    return risco + pen

# ============================================================
# 7. IMPLEMENTAÇÃO DO BRKGA
# ============================================================

POP_SIZE = 2000     # tamanho total da população inicial do BRKGA (quantos indivíduos existem por geração)
ELITE_RATE = 0.1   # fração da população que será considerada elite (10%)
MUTANT_RATE = 0.5  # fração da população que será substituída por indivíduos aleatórios (mutantes)
RHO = 0.9          # probabilidade de herdar o gene do pai ELITE no crossover (tendência a copiar o elite)
N_GENERATIONS = 2000  # número total de gerações do algoritmo

# espaço vazio para o filho child (, ,  , ,)
mask = np.random.rand(n) < RHO   # cria máscara booleana: True = herda do elite, False = herda do não-elite
                                 # exemplo: [True, False, True] → gene 1 do elite, gene 2 do não elite, gene 3 do elite

child[mask] = elite[mask]        # copia genes do pai elite onde mask = True
child[~mask] = non_elite[~mask]  # copia genes do pai não elite onde mask = False
return child                     # retorna o filho completo

# Inicializa população
population = [gerar_individuo() for _ in range(POP_SIZE)]  
# cria POP_SIZE indivíduos totalmente aleatórios como população inicial

# Loop de gerações
best_overall = None                      # melhor indivíduo global encontrado até agora
best_fitness_overall = float("inf")      # melhor fitness (menor valor) encontrado até agora

for g in range(N_GENERATIONS):           # executa o processo evolutivo por várias gerações
    # Avaliar população
    fitness_values = [fitness(ind) for ind in population]  
    # calcula o fitness de cada indivíduo

    # Ordenar por fitness
    idx_sorted = np.argsort(fitness_values)  
    # retorna índices ordenados do menor fitness (melhor) para o maior

    population = [population[i] for i in idx_sorted]  
    # reorganiza população da melhor para a pior

    fitness_values = [fitness_values[i] for i in idx_sorted]  
    # reorganiza também os valores de fitness

    # Atualiza melhor global
    if fitness_values[0] < best_fitness_overall:  
        # se o melhor da geração é melhor que o melhor da história…
        best_fitness_overall = fitness_values[0]      # atualiza melhor fitness global
        best_overall = population[0].copy()           # guarda o melhor indivíduo

    if (g+1) % 20 == 0 or g == 0:
        # imprime a cada 20 gerações (ou na primeira)
        print(f"Geração {g+1}: melhor fitness = {fitness_values[0]:.6f}")

    # Separa elites e não-elites
    elites = population[:N_ELITE]          # pega os melhores N_ELITE indivíduos
    non_elites = population[N_ELITE:]      # o resto é não-elite

    # Nova população
    new_population = []                    # começa uma nova lista para a próxima geração

    # 1) Elites vão direto
    new_population.extend(elites)          # elites passam para a próxima geração sem mudança

    # 2) Offsprings por crossover enviesado
    for _ in range(N_OFFSPRING):           # gera filhos através do crossover enviesado
        elite_parent = random.choice(elites)          # escolhe 1 pai elite
        non_elite_parent = random.choice(non_elites)  # escolhe 1 pai não elite
        child = crossover(elite_parent, non_elite_parent)  # cria filho enviesado
        new_population.append(child)                   # adiciona filho na população nova

    # 3) Mutantes (indivíduos aleatórios)
    for _ in range(N_MUTANT):              # adiciona mutantes aleatórios
        new_population.append(gerar_individuo())

    population = new_population            # atualiza população final da geração

print("\n=== RESULTADO FINAL ===")
print(f"Melhor fitness encontrado: {best_fitness_overall:.6f}")  
# exibe fitness do melhor indivíduo encontrado

# Decodifica melhor solução
w_best, b_best, met_best = decoder(best_overall)  
# transforma o cromossomo em pesos, seleção e métricas

# Converte retorno e risco para percentual
ret_percent = met_best['retorno'] * 100        # retorno anual em %
vol_percent = np.sqrt(met_best['risco']) * 100 # risco anual (volatilidade) em %

print(f"Retorno da carteira: {ret_percent:.2f}%")
print(f"Risco (volatilidade): {vol_percent:.2f}%")
print(f"Nº de setores:        {met_best['n_total']}")

setores_ativos = [ticker_to_sector[tickers[i]] for i in range(n) if b_best[i] == 1]  
# lista os setores presentes apenas dos ativos escolhidos

print("Setores presentes:", sorted(set(setores_ativos)))  
# imprime setores únicos escolhidos

# Mostra ativos selecionados
print("\nAtivos selecionados:")
for i in range(n):
    if b_best[i] == 1:   # se ativo foi escolhido
        print(f"{tickers[i]:<10}  w = {w_best[i]*100:>6.2f}%  "
              f"setor = {ticker_to_sector[tickers[i]]:<20}  "
              f"mu = {mu_vec[i]*100:6.2f}%")
        # imprime: ticker, peso em %, setor e retorno anual em %
