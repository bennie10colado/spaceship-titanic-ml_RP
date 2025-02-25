from src.data_loader import load_data
from src.models.knn_model import preprocess_data, train_knn
from src.models.neural_network import train_mlp
import argparse


def main():
    parser = argparse.ArgumentParser(description="Script de treinamento e avaliação de modelo")

    parser.add_argument('--train', action='store_true', help="Treinar o modelo")
    parser.add_argument('--batch_size', type=int, default=2, help="Tamanho de batch")
    parser.add_argument('--epochs', type=int, default=5, help="Número de épocas para o treinamento")
    parser.add_argument('--model_path', type=str, default="weights/default.h5", help="Caminho do modelo (para treinar e carregar)")

    args = parser.parse_args()

    # carregar os dados
    train_df, test_df = load_data()

    if train_df is None:
        print("❌ Erro ao carregar os dados. Encerrando execução.")
        return

    # pré-processar os dados
    X, y, scaler, label_encoders = preprocess_data(train_df)

    # treinar e avaliar o modelo k-NN
    accuracy_knn, knn_model = train_knn(X, y, k=10)
    accuracy_mlp, mlp_model = train_mlp(X, y, args)

if __name__ == "__main__":
    main()
    #TODO: podemos fazer umas tabelas com matplotlib posteriormente como a matriz de confusão, e curva roc...
