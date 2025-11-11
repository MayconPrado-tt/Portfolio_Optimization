import pandas as pd
import requests


def gerar_arquivo_mapeamento_setores():
    """
    Obtém a classificação setorial do S&P 500 da Wikipedia e salva em um arquivo CSV.
    """
    try:
        print("Obtendo mapeamento de setores da Wikipedia...")
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        tabela = pd.read_html(response.text)
        df_sp500 = tabela[0]

        # Corrige o ticker (ex: BRK.B -> BRK-B)
        df_sp500['Symbol'] = df_sp500['Symbol'].str.replace(
            '.', '-', regex=False)

        # Seleciona apenas as colunas que nos interessam
        df_mapeamento = df_sp500[['Symbol', 'GICS Sector']].copy()
        df_mapeamento.rename(
            columns={'Symbol': 'Ticker', 'GICS Sector': 'Setor'}, inplace=True)

        # Salva o arquivo
        df_mapeamento.to_csv('mapeamento_setores.csv', index=False)

        print(
            f"Sucesso! Mapeamento para {len(df_mapeamento)} tickers salvo em 'mapeamento_setores.csv'.")
        return True

    except Exception as e:
        print(f"ERRO ao gerar arquivo de mapeamento de setores: {e}")
        return False


if __name__ == '__main__':
    gerar_arquivo_mapeamento_setores()
