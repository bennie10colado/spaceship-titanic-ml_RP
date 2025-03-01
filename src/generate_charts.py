import matplotlib.pyplot as plt
import numpy as np
import os

PATH = "src/charts"
os.makedirs(PATH, exist_ok=True)

def generate_precision_chart(accuracies_knn, accuracies_mlp):
    """
    Gera um gráfico de barras comparando a precisão média dos classificadores k-NN e MLP.

    Parâmetros:
        accuracies_knn (array-like): Acurácias obtidas para o classificador k-NN.
        accuracies_mlp (array-like): Acurácias obtidas para o classificador MLP.
    """
    mean_knn = np.mean(accuracies_knn)
    mean_mlp = np.mean(accuracies_mlp)

    models = ['k-NN', 'MLP']
    means = [mean_knn, mean_mlp]

    plt.figure(figsize=(6,4))
    bars = plt.bar(models, means, color=['blue', 'green'])
    plt.xlabel('Modelos')
    plt.ylabel('Precisão Média')
    plt.title('Comparação de Precisão Média dos Modelos')
    plt.ylim(0,1)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.4f}", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PATH, "precision_comparison.png"))
    plt.show()


if __name__ == "__main__":
    
#Resultados do 10x Holdout 50/50:
# - k-NN: [0.77547734 0.77478721 0.77340695 0.77616747 0.77846791 0.77846791
# 0.77639752 0.77593743 0.77202669 0.76535542]
# - MLP:  [0.79411088 0.79710145 0.78490913 0.79273062 0.79250058 0.78743961
# 0.79135036 0.79779158 0.79572119 0.79595123]
 
 
    acc_knn_example = [0.77547734, 0.77478721, 0.77340695, 0.77616747, 0.77846791, 0.77846791,
 0.77639752, 0.77593743, 0.77202669, 0.76535542]
    acc_mlp_example = [0.79411088, 0.79710145, 0.78490913, 0.79273062, 0.79250058, 0.78743961,
 0.79135036, 0.79779158, 0.79572119, 0.79595123]
    
    #TODO: podemos posteriormente fazer mais umas tabelas com matplotlib posteriormente como a matriz de confusão, e curva roc...
    generate_precision_chart(acc_knn_example, acc_mlp_example)
