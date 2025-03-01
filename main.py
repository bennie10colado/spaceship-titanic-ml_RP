from generate_charts import generate_precision_chart
from src.data_loader import load_data
from src.models.knn_model import preprocess_data, train_knn
from src.models.neural_network import train_mlp
import argparse

from src.evaluation import (
    repeated_holdout,
    holdout_10x_50_50,
    paired_t_test,
    confidence_interval_diff,
    confidence_interval,
    check_interval_overlap,
    evaluate_knn_split,
    evaluate_mlp_split
)

def main():
    parser = argparse.ArgumentParser(description="Script de treinamento e avaliação de modelo")
    parser.add_argument('--train', action='store_true', help="Treinar o modelo")
    parser.add_argument('--batch_size', type=int, default=2, help="Tamanho de batch")
    parser.add_argument('--epochs', type=int, default=5, help="Número de épocas para o treinamento")
    parser.add_argument('--model_path', type=str, default="weights/default.h5", help="Caminho do modelo (para treinar e carregar)")
    args = parser.parse_args()

    # Carregar os dados
    train_df, test_df = load_data()
    if train_df is None:
        print("❌ Erro ao carregar os dados. Encerrando execução.")
        return

    # Pré-processar os dados
    X, y, scaler, label_encoders = preprocess_data(train_df)

    # Treinar e avaliar o modelo k-NN (holdout simples)
    accuracy_knn, knn_model = train_knn(X, y, k=10)
    # Treinar e avaliar o modelo MLP (holdout simples)
    accuracy_mlp, mlp_model = train_mlp(X, y, args)

    # ---------------------------
    # Avaliação Repetida (Holdout Aleatório Repetido)
    # ---------------------------
    n_repeats = 30
    acc_knn = repeated_holdout(evaluate_knn_split, X, y, n_repeats=n_repeats, test_size=0.3, random_state=42, k=10)
    acc_mlp = repeated_holdout(evaluate_mlp_split, X, y, n_repeats=n_repeats, test_size=0.3,
                                 random_state=27, batch_size=args.batch_size, epochs=args.epochs)

    print(f"\nResultados do Holdout Aleatório Repetido (n={n_repeats}):")
    print(f" - k-NN: {acc_knn}")
    print(f" - MLP:  {acc_mlp}")

    # ---------------------------
    # 10x Holdout 50/50
    # ---------------------------
    acc_knn_10x = holdout_10x_50_50(evaluate_knn_split, X, y, n_repeats=10, random_state=42, k=10)
    acc_mlp_10x = holdout_10x_50_50(evaluate_mlp_split, X, y, n_repeats=10, random_state=27,
                                    batch_size=args.batch_size, epochs=args.epochs)

    print(f"\nResultados do 10x Holdout 50/50:")
    print(f" - k-NN: {acc_knn_10x}")
    print(f" - MLP:  {acc_mlp_10x}")

    # ---------------------------
    # Teste de Hipótese (Teste t pareado)
    # ---------------------------
    t_stat, p_value, significant = paired_t_test(acc_knn, acc_mlp)
    print(f"\nTeste t pareado entre k-NN e MLP:")
    print(f" t-statística: {t_stat:.4f}, p-value: {p_value:.4f}")
    print(f" Diferença estatisticamente significativa? {significant}")

    # ---------------------------
    # Intervalos de Confiança
    # ---------------------------
    ci_diff = confidence_interval_diff(acc_knn, acc_mlp)
    ci_knn = confidence_interval(acc_knn)
    ci_mlp = confidence_interval(acc_mlp)
    print(f"\nIntervalo de Confiança para a diferença das médias: {ci_diff}")
    print(f"Intervalo de Confiança k-NN: {ci_knn}")
    print(f"Intervalo de Confiança MLP: {ci_mlp}")

    # ---------------------------
    # Verificação de Sobreposição dos Intervalos de Confiança
    # ---------------------------
    overlap = check_interval_overlap(ci_knn, ci_mlp)
    print(f"\nOs intervalos de confiança se sobrepõem? {overlap}")
    
    # Gerar gráfico comparativo da precisao media dos modelos
    generate_precision_chart(acc_knn, acc_mlp)
    
    
if __name__ == "__main__":
    main()
