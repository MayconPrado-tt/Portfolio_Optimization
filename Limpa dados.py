import pandas as pd
import numpy as np

ARQUIVO_ENTRADA = 'precos_fechamento_3_anos.csv'

LIMITE_DADOS_MINIMOS = 0.95

DIAS_UTEIS_ANO = 252

ARQUIVO_SAIDA_MU = 'vetor_retornos_calculado.csv'
ARQUIVO_SAIDA_SIGMA = 'matriz_covariancia_calculada.csv'

def processar_arquivo_local(arquivo_entrada):
    """
    Carrega o arquivo CSV local, limpa os dados e calcula os parâmetros
    de retorno esperado (mu) e covariância (Sigma).
    """
    try:
        # --- Leitura e Limpeza Inicial ---
        print(f"Lendo dados do arquivo '{arquivo_entrada}'...")
        # A primeira coluna  é a data
        df_precos = pd.read_csv(arquivo_entrada, index_col=0, parse_dates=True)
        print(f"Arquivo carregado com sucesso. {df_precos.shape[1]} ativos encontrados.")

        # --- Filtro de Qualidade dos Dados ---
        minimo_pontos_dados = int(len(df_precos) * LIMITE_DADOS_MINIMOS)
        print(f"Removendo ativos com menos de {minimo_pontos_dados} dias de dados...")
        df_precos_filtrado = df_precos.dropna(axis='columns', thresh=minimo_pontos_dados)
        
        num_removidos = df_precos.shape[1] - df_precos_filtrado.shape[1]
        print(f"{num_removidos} ativos foram removidos.")
        print(f"{df_precos_filtrado.shape[1]} ativos permanecerão na análise.")
        
        # Preenche pequenas lacunas restantes com o último preço válido
        df_precos_limpo = df_precos_filtrado.fillna(method='ffill').dropna(axis='rows')

        # --- Cálculo dos Parâmetros ---
        print("\nCalculando retornos diários...") # aplica a formula (valor_hoje - valor_ontem) / valor_ontem para cada celula
        retornos_diarios = df_precos_limpo.pct_change().dropna()
        
        print("Calculando o vetor de retornos esperados (μ) e a matriz de covariância (Σ)...")
        mu_anual = retornos_diarios.mean() * DIAS_UTEIS_ANO
        Sigma_anual = retornos_diarios.cov() * DIAS_UTEIS_ANO
        
        # --- Salvamento dos Resultados ---
        print(f"Salvando resultados em arquivos CSV...")
        mu_anual.to_csv(ARQUIVO_SAIDA_MU, header=['retorno_anual'], decimal='.')
        Sigma_anual.to_csv(ARQUIVO_SAIDA_SIGMA, decimal='.')
        
        print("\n" + "="*50)
        print("PROCESSO CONCLUÍDO COM SUCESSO!")
        print("="*50)
        print(f"Vetor de retornos (μ) salvo em: '{ARQUIVO_SAIDA_MU}'")
        print(f"Matriz de covariância (Σ) salva em: '{ARQUIVO_SAIDA_SIGMA}'")
        print(f"\nPrimeiras 5 linhas do vetor de retornos (μ):")
        print(mu_anual.head())

    except FileNotFoundError:
        print(f"ERRO: Arquivo de entrada '{arquivo_entrada}' não encontrado.")
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")

if __name__ == '__main__':
    processar_arquivo_local(ARQUIVO_ENTRADA)