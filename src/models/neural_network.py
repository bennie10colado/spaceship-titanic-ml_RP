import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from src.data_loader import load_data
from tensorflow.keras.models import load_model
from src.models.knn_model import preprocess_data
from sklearn.metrics import accuracy_score

def create_model(num_features):
    # Criando um modelo simples
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(num_features,)),
        layers.Dropout(0.3),  
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')  # Saída binária
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_mlp(X_train, y_train, X_val, y_val, args):
    """
    Treina e avalia um modelo MLP.

    Parâmetros:
        X (array): Features normalizadas.
        y (array): Labels.
        test_size (float): Percentual dos dados usados para validação.

    Retorna:
        accuracy (float): Acurácia do modelo.
        model (MLP): Modelo treinado.
    """
    # divisão holdout
    #X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=27, stratify=y)


    if args.train:
        model = create_model(X_train.shape[1])

        # treinar o modelo k-NN
        model.fit(x=X_train, y=y_train, batch_size=args.batch_size, epochs=args.epochs)
        model.save(args.model_path)
        print(f"Modelo salvo em {args.model_path}")
    else:
        model = load_model(args.model_path)
        print(f"Modelo carregado de {args.model_path}")
        

    # faz as previsões
    y_pred_proba = model.predict(X_val)
    y_pred = (y_pred_proba > 0.5).astype(int)

    # avalia o modelo
    accuracy = accuracy_score(y_val, y_pred)
    print(f"📊 Acurácia do MLP: {accuracy:.4f}")

    return accuracy, model
