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
    Aplica pré-processamento ao dataset.

    Parâmetros:
        df (pd.DataFrame): DataFrame a ser pré-processado.
        is_train (bool): Indica se é conjunto de treino (para manter `y`).

    Retorna:
        X (pd.DataFrame): Features processadas.
        y (pd.Series) ou None: Target processado (se `is_train=True`).
        scaler (StandardScaler): Objeto de normalização para reutilização.
        label_encoders (dict): Encoders utilizados para variáveis categóricas.
    """
    # Remover colunas irrelevantes
    df.drop(["PassengerId", "Name"], axis=1, inplace=True, errors="ignore")

    # Separar colunas categóricas e numéricas
    categorical_cols = df.select_dtypes(include=["object"]).columns
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns

    # Criar imputadores para valores ausentes
    imputer_categorical = SimpleImputer(strategy="most_frequent")  # Preenche com a moda
    imputer_numerical = SimpleImputer(strategy="median")  # Preenche com a mediana

    # Aplicar imputação
    df[categorical_cols] = imputer_categorical.fit_transform(df[categorical_cols])
    df[numerical_cols] = imputer_numerical.fit_transform(df[numerical_cols])

    # Codificar colunas categóricas
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Separar features e target
    if is_train:
        X = df.drop("Transported", axis=1)
        y = df["Transported"]
    else:
        X = df
        y = None

    # Normalizar os dados
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, scaler, label_encoders

def train_knn(X, y, k=5, test_size=0.2):
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
    # Divisão holdout
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    # Treinar modelo k-NN
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    # Fazer previsões
    y_pred = model.predict(X_val)

    # Avaliar o modelo
    accuracy = accuracy_score(y_val, y_pred)
    print(f"📊 Acurácia do k-NN (k={k}): {accuracy:.4f}")

    return accuracy, model

# Teste do módulo (remova caso use como import)
if __name__ == "__main__":
    # Carregar dados
    train_df, _ = load_data()

    # Pré-processar os dados
    X, y, scaler, label_encoders = preprocess_data(train_df)

    # Treinar k-NN
    accuracy, knn_model = train_knn(X, y, k=5)
