import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy import stats
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def evaluate_knn_split(X_train, y_train, X_test, y_test, k=10):
    """
    Executa treinamento e avaliação do k-NN a partir de dados já divididos.
    """
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, model

def evaluate_mlp_split(X_train, y_train, X_test, y_test, batch_size=2, epochs=5):
    """
    Executa treinamento e avaliação de uma rede neural (MLP) a partir de dados já divididos.
    """
    def create_model(num_features):
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(num_features,)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    model = create_model(X_train.shape[1])
    # Treinamento silencioso (verbose=0)
    model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=0)
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, model

def repeated_holdout(model_func, X, y, n_repeats=10, test_size=0.3, random_state=None, **kwargs):
    """
    Executa holdout aleatório repetido.
    
    Parâmetros:
        model_func: função que recebe (X_train, y_train, X_test, y_test) e retorna (accuracy, model).
        n_repeats: número de repetições.
        test_size: fração dos dados para teste.
        random_state: semente para reprodução.
        **kwargs: parâmetros adicionais (por exemplo, k para k-NN ou batch_size/epochs para MLP).
    
    Retorna:
        Um array de acurácias obtidas em cada repetição.
    """
    accuracies = []
    for i in range(n_repeats):
        rs = random_state + i if random_state is not None else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=rs, stratify=y)
        acc, _ = model_func(X_train, y_train, X_test, y_test, **kwargs)
        accuracies.append(acc)
    return np.array(accuracies)

def holdout_10x_50_50(model_func, X, y, n_repeats=10, random_state=None, **kwargs):
    """
    Executa 10 repetições de holdout com divisão fixa de 50% para treino e 50% para teste.
    """
    return repeated_holdout(model_func, X, y, n_repeats=n_repeats, test_size=0.5,
                              random_state=random_state, **kwargs)

def paired_t_test(acc1, acc2, alpha=0.05):
    """
    Realiza o teste t pareado entre duas séries de acurácias.
    
    Retorna:
        t_stat: estatística t
        p_value: valor p
        significant: True se p_value < alpha
    """
    t_stat, p_value = stats.ttest_rel(acc1, acc2)
    significant = p_value < alpha
    return t_stat, p_value, significant

def confidence_interval_diff(acc1, acc2, confidence=0.95):
    """
    Calcula o intervalo de confiança para a diferença entre as médias de duas séries de acurácias.
    """
    differences = acc1 - acc2
    mean_diff = np.mean(differences)
    sem_diff = stats.sem(differences)
    n = len(differences)
    t_val = stats.t.ppf((1 + confidence) / 2.0, n - 1)
    margin = t_val * sem_diff
    lower = mean_diff - margin
    upper = mean_diff + margin
    return lower, upper

def confidence_interval(acc, confidence=0.95):
    """
    Calcula o intervalo de confiança para a acurácia média de um classificador.
    """
    n = len(acc)
    mean_acc = np.mean(acc)
    sem = stats.sem(acc)
    t_val = stats.t.ppf((1 + confidence) / 2.0, n - 1)
    margin = t_val * sem
    return mean_acc - margin, mean_acc + margin

def check_interval_overlap(ci1, ci2):
    """
    Verifica se dois intervalos de confiança (ci1 e ci2) se sobrepõem.
    """
    return max(ci1[0], ci2[0]) <= min(ci1[1], ci2[1])
