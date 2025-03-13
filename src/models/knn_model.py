from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import os
from src.data_loader import load_data

def preprocess_data(df, is_train=True):
    """
    Aplica pré-processamento dos dados ao dataset.

    Parâmetros:
        df (pd.DataFrame): DataFrame a ser pré-processado.
        is_train (bool): Indica se é conjunto de treino (para manter `y`).

    Retorna:
        X (pd.DataFrame): Features processadas.
        y (pd.Series) ou None: Target processado (se `is_train=True`).
        scaler (StandardScaler): Objeto de normalização para reutilização.
        label_encoders (dict): Encoders utilizados para variáveis categóricas.
    """
    # remover as colunas irrelevantes
    df.drop(["PassengerId", "Name"], axis=1, inplace=True, errors="ignore")

    # separar colunas categoricas e numericas
    categorical_cols = df.select_dtypes(include=["object"]).columns
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns

    # criar imputadores para valores ausentes
    imputer_categorical = SimpleImputer(strategy="most_frequent")  # preenche cat. com a moda
    imputer_numerical = SimpleImputer(strategy="median")  # preenche num. com a mediana

    # aplicar imputação
    df[categorical_cols] = imputer_categorical.fit_transform(df[categorical_cols])
    df[numerical_cols] = imputer_numerical.fit_transform(df[numerical_cols])

    #TODO: adicionar imputação no holdout de treino, com os valores somente da base de treino, e não utilizar dados da validação
    # codificar colunas categóricas
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    #TODO: ver exatamente quantas possibilidades de labeled enconding temos nos atributos categoricos
    #TODO: ver extamente os valores das cabines, ver como se comportam os valores, e como as caracteristicas estão presentes nos dados antes de enviar para a mlp, e no knn. Printar as colunas, e os dados
    #TODO: COLOCAR NO RELATÒRIO COMO TRATOU CADA COLUNA
    # separando features e target
    if is_train:
        X = df.drop("Transported", axis=1)
        y = df["Transported"]
    else:
        X = df
        y = None

    # normaliza os dados
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, scaler, label_encoders

def train_knn(X, y, k=20, test_size=0.1):
    """
    Treina e avalia um modelo k-NN com holdout.

    Parâmetros:
        X (array): Features normalizadas.
        y (array): Labels.
        k (int): Número de vizinhos do k-NN.
        test_size (float): Percentual dos dados usados para validação.

    Retorna:
        accuracy (float): Acurácia do modelo.
        model (KNeighborsClassifier): Modelo treinado.
    """
    # divisão holdout
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    #TODO: adicionar o preprocessamento do treino, de valores invalidos/null com media/moda 
    # TODO: treinar o modelo k-NN, somente com os dados da base de treino
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    # faz as previsões
    y_pred = model.predict(X_val)

    # avalia o modelo
    accuracy = accuracy_score(y_val, y_pred)
    print(f"📊 Acurácia do k-NN (k={k}): {accuracy:.4f}")

    return accuracy, model

# teste do arquivo individualmente
if __name__ == "__main__":
    train_df, _ = load_data()
    X, y, scaler, label_encoders = preprocess_data(train_df)
    accuracy, knn_model = train_knn(X, y, k=5)
