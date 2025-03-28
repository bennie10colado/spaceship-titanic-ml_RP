from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import os
from src.data_loader import load_data
from sklearn.preprocessing import OneHotEncoder

def preprocess_data(df, test_size=0.1):
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
    
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df["Transported"])

    # separar colunas categoricas e numericas
    categorical_cols = train_df.select_dtypes(include=["object"]).columns
    numerical_cols = train_df.select_dtypes(include=["int64", "float64"]).columns

    # criar imputadores para valores ausentes
    imputer_categorical = SimpleImputer(strategy="most_frequent")  # preenche cat. com a moda
    imputer_numerical = SimpleImputer(strategy="median")  # preenche num. com a mediana

    # aplicar imputação no conjunto de treino
    train_df[categorical_cols] = imputer_categorical.fit_transform(train_df[categorical_cols])
    train_df[numerical_cols] = imputer_numerical.fit_transform(train_df[numerical_cols])
    
    # aplicar imputação no conjunto de validaçao treino
    val_df[categorical_cols] = imputer_categorical.transform(val_df[categorical_cols])
    val_df[numerical_cols] = imputer_numerical.transform(val_df[numerical_cols])


    # exibir informações detalhadas das colunas categóricas para o relatório
    for col in categorical_cols:
        unique_vals = train_df[col].unique()
        print(f"Coluna '{col}' - Número de categorias: {len(unique_vals)} - Valores: {unique_vals}")
        if col.lower() == "cabin":
            sorted_vals = np.sort(unique_vals)
            print(f"Valores ordenados da coluna 'Cabin': {sorted_vals}")
    
    print("Resumo do tratamento das colunas: valores ausentes foram imputados e codificação OneHot aplicada nas colunas categóricas.")
    
    
    # codificar colunas categóricas c OneHotEncoder 
    
    ohe = OneHotEncoder(handle_unknown='ignore',sparse_output=False)
    train_cat = ohe.fit_transform(train_df[categorical_cols])
    val_cat = ohe.transform(val_df[categorical_cols])
    
    # Converter os arrays resultantes para DataFrame com nomes das colunas certo
    train_cat_df = pd.DataFrame(train_cat, index=train_df.index, columns=ohe.get_feature_names_out(categorical_cols))
    val_cat_df = pd.DataFrame(val_cat, index=val_df.index, columns=ohe.get_feature_names_out(categorical_cols))

    # Remover as colunas originais e concatenar os dados codificados
    train_df = train_df.drop(columns=categorical_cols)
    val_df = val_df.drop(columns=categorical_cols)
    train_df = pd.concat([train_df, train_cat_df], axis=1)
    val_df = pd.concat([val_df, val_cat_df], axis=1)
    

    # codificar colunas categóricas OLD
    #label_encoders = {}
    #for col in categorical_cols:
    #   le = LabelEncoder()
    #   train_df[col] = le.fit_transform(train_df[col])
    #   val_df[col] = le.transform(val_df[col])
    #   label_encoders[col] = le
    #TODO: ver exatamente quantas possibilidades de labeled enconding temos nos atributos categoricos
    #TODO: ver extamente os valores das cabines, ver como se comportam os valores, e como as caracteristicas estão presentes nos dados antes de enviar para a mlp, e no knn. Printar as colunas, e os dados
    #TODO: COLOCAR NO RELATÒRIO COMO TRATOU CADA COLUNA
    
    
    # separando features e target
    X_train = train_df.drop("Transported", axis=1)
    y_train = train_df["Transported"]
    X_val = val_df.drop("Transported", axis=1)
    y_val = val_df["Transported"]

    # normaliza os dados
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    return X_train, y_train, X_val, y_val, scaler, ohe

def train_knn(X_train, y_train, X_val, y_val, k=20):
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
    
    # verificar e imputar valores faltantes, se existirem (apenas como precaução)
    if np.isnan(X_train).any():
        print("Encontrados valores faltantes em X_train. Aplicando imputação com média.")
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        X_train = imputer.fit_transform(X_train)

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
    X_train, y_train, X_val, y_val, scaler, label_encoders = preprocess_data(train_df, test_size=0.1)
    accuracy, knn_model = train_knn(X_train, y_train, X_val, y_val, k=5)
