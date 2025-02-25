import numpy as np

class ManualKNNClassifier:
    """
    implementacao manual do algoritmo k-nearest neighbors (k-nn) para classificacao.

    o k-nn e um metodo baseado em instancias que classifica uma nova amostra com base
    na votacao majoritaria dos 'k' vizinhos mais proximos, usando a distancia euclidiana.
    """
    def __init__(self, x_train, y_train, k):
        """
        inicializa o classificador com os dados de treino e o numero de vizinhos.

        parametros:
            x_train (np.array): array com as features do conjunto de treino.
            y_train (np.array): array com os rotulos correspondentes.
            k (int): numero de vizinhos para a votacao.
        """
        self.x_train = x_train
        self.y_train = y_train
        self.k = k

    def predict(self, x_test):
        """
        realiza a previsao das classes para cada amostra em x_test.

        para cada amostra de teste, calcula a distancia euclidiana ate todas as amostras
        de treino, seleciona os 'k' vizinhos mais proximos e define a classe pela votacao.

        parametros:
            x_test (np.array): array com as features das amostras de teste.

        retorna:
            np.array: array com a classe predita para cada amostra.
        """
        predictions = []  # lista para armazenar as previsoes

        for sample in x_test:
            # calcula a distancia euclidiana entre a amostra e cada amostra de treino
            distances = np.sqrt(np.sum((self.x_train - sample) ** 2, axis=1))
            # obtem os indices dos k vizinhos mais proximos
            nearest_indices = np.argsort(distances)[:self.k]
            # obtem os rotulos dos k vizinhos
            neighbor_labels = self.y_train[nearest_indices]
            # votacao: conta a frequencia de cada rotulo e seleciona o de maior contagem
            unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
            predicted_label = unique_labels[np.argmax(counts)]
            predictions.append(predicted_label)
        
        return np.array(predictions)

def train_knn_manual(x_data, y, k=10, test_size=0.1, random_state=42):
    """
    treina e avalia o classificador k-nn usando holdout.

    passos:
      1. embaralha os indices dos dados.
      2. divide os dados em treino e validacao.
      3. treina o classificador com os dados de treino.
      4. faz a previsao e calcula a acuracia no conjunto de validacao.

    parametros:
      x_data (np.array): array com todas as features.
      y (np.array): array com os rotulos.
      k (int): numero de vizinhos.
      test_size (float): proporcao dos dados para validacao.
      random_state (int): semente para reproducibilidade.

    retorna:
      accuracy (float): acuracia no conjunto de validacao.
      model (ManualKNNClassifier): modelo treinado.
    """
    # cria um gerador de numeros aleatorios com semente
    rng = np.random.default_rng(random_state)
    # embaralha os indices
    indices = rng.permutation(len(x_data))
    
    # calcula o indice de divisao
    split_index = int((1 - test_size) * len(x_data))
    
    # divide os indices em treino e validacao
    train_indices = indices[:split_index]
    val_indices = indices[split_index:]
    
    # separa os dados
    x_train = x_data[train_indices]
    x_val = x_data[val_indices]
    y_train = y[train_indices]
    y_val = y[val_indices]
    
    # cria o modelo k-nn com os dados de treino
    model = ManualKNNClassifier(x_train, y_train, k)
    
    # realiza a previsao no conjunto de validacao
    y_pred = model.predict(x_val)
    
    # calcula a acuracia comparando as previsoes com os rotulos reais
    accuracy = np.mean(y_pred == y_val)
    print(f"📊 acuracia do k-nn (k={k}): {accuracy:.4f}")
    
    return accuracy, model

if __name__ == "__main__":
    # cria um gerador de numeros aleatorios para reproducibilidade
    rng = np.random.default_rng(42)
    
    # gera 100 amostras com 5 features cada
    x_data = rng.random((100, 5))
    
    # gera 100 rotulos binarios (0 ou 1)
    y = rng.integers(0, 2, 100)
    
    # treina o classificador k-nn com k=10 e exibe a acuracia
    accuracy, knn_model = train_knn_manual(x_data, y, k=10)
