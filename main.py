from src.data_loader import load_data
from src.models.knn_model import preprocess_data, train_knn

def main():
    # carregar os dados
    train_df, test_df = load_data()

    if train_df is None:
        print("❌ Erro ao carregar os dados. Encerrando execução.")
        return

    # pré-processar os dados
    X, y, scaler, label_encoders = preprocess_data(train_df)

    # treinar e avaliar o modelo k-NN
    accuracy, knn_model = train_knn(X, y, k=10)

if __name__ == "__main__":
    main()
    #TODO: podemos fazer umas tabelas com matplotlib posteriormente como a matriz de confusão, e curva roc...