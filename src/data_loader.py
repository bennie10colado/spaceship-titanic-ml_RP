import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

def load_data():
    """
    Carrega e trata os datasets de treino e teste.

    Retorna:
        train_df (pd.DataFrame): DataFrame de treino com `Transported` convertido para 0/1.
        test_df (pd.DataFrame): DataFrame de teste para avaliação.
    """
    try:
        # carregar os dados
        train_df = pd.read_csv(TRAIN_PATH)
        test_df = pd.read_csv(TEST_PATH)

        # converter coluna target para numérico
        train_df["Transported"] = train_df["Transported"].astype(int)

        # exibir resumo dos dados carregados
        print(f"✅ Dados carregados com sucesso!")
        print(f"🔹 Treino: {train_df.shape} | Teste: {test_df.shape}")
        
        return train_df, test_df

    except FileNotFoundError as e:
        print(f"❌ Erro ao carregar os arquivos: {e}")
        return None, None

# teste do arquivo individualmente
if __name__ == "__main__":
    train, test = load_data()
    print(train.head(), test.head())
