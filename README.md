# Spaceship Titanic ML 🚀

Este repositório contém uma solução para o problema do **Spaceship Titanic** no Kaggle, utilizando **k-NN** para classificação. O projeto está estruturado de forma modular para facilitar futuras implementações, como redes neurais.

## 📌 Pré-requisitos

Antes de começar, você precisa ter:

- **Windows com WSL (Windows Subsystem for Linux) ativado** (ou rodar diretamente em Linux/Mac)
- **Python 3.8+** instalado
- **Git** instalado

## 📂 Estrutura do Projeto

```
spaceship-titanic-ml/
│── data/                  # Pasta com os dados (train.csv, test.csv, etc.)
│── src/                   # Código-fonte do projeto
│   │── data_loader.py      # Carregamento dos dados
│   │── preprocessing.py    # Pré-processamento dos dados
│   │── evaluation.py       # Funções para avaliação de modelos
│   └── models/             # Modelos de Machine Learning
│       │── knn_model.py    # Implementação do modelo k-NN
│       │── neural_network.py # (Futuro) Implementação de redes neurais
│── venv/                   # Ambiente virtual Python
│── main.py                 # Arquivo principal para rodar o pipeline
│── requirements.txt         # Dependências do projeto
│── README.md                # Documentação do projeto
```

---

## ⚙️ **Configuração do Ambiente**

### 1️⃣ **Ativar o WSL (se estiver no Windows)**
Se estiver no **Windows**, certifique-se de que o **WSL (Windows Subsystem for Linux)** está ativado. Para instalar o WSL, execute no **PowerShell (como Administrador)**:

```powershell
wsl --install
```
Após a instalação, reinicie o PC e abra o **Ubuntu** pelo WSL.

---

### 2️⃣ **Clonar o Repositório**

Abra o terminal (WSL/Linux/Mac) e clone este repositório:

```bash
git clone https://github.com/SEU-USUARIO/spaceship-titanic-ml.git
cd spaceship-titanic-ml
```

---

### 3️⃣ **Criar e Ativar o Ambiente Virtual**

```bash
# Criar ambiente virtual
python3 -m venv venv

# Ativar o ambiente virtual
source venv/bin/activate  # Linux/Mac/WSL
```

---

### 4️⃣ **Instalar Dependências**

Com o ambiente virtual ativado, instale as dependências:

```bash
pip install -r requirements.txt
```

---

## 🚀 **Rodando o Projeto**

Com tudo configurado, execute:

```bash
python3 main.py
```

Isso irá:

1. **Carregar os dados**
2. **Pré-processar os dados**
3. **Treinar o modelo k-NN**
4. **Exibir a acurácia do modelo**

---


## 📜 **Próximos Passos**
✅ Implementação de Redes Neurais (próxima fase).  
✅ Testes com Validação Cruzada.  
✅ Comparação com outros modelos como Random Forest e SVM.

---

