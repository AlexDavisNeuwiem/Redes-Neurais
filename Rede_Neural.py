import numpy as np
import h5py
import matplotlib.pyplot as plot
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Função para carregar o conjunto de dados a partir de um arquivo HDF5
def carregarConjuntoDeDadosDeArquivo(nome_arquivo):
    data = h5py.File(nome_arquivo, 'r')
    # Verifica se o conjunto de treinamento ou teste está no arquivo
    if 'train_set_x' in data:
        dataX = np.array(data['train_set_x'][:])  # Dados de entrada (imagens)
        dataY = np.array(data['train_set_y'][:])  # Rótulos de saída
    elif 'test_set_x' in data:
        dataX = np.array(data['test_set_x'][:])  # Dados de entrada (imagens)
        dataY = np.array(data['test_set_y'][:])  # Rótulos de saída
    else:
        print("Arquivo inesperado\nEncerrando...")
        quit()
    return dataX, dataY

# Função para normalizar os dados dividindo os valores dos pixels por 255
# Isso converte os valores de pixels de 0-255 para o intervalo 0-1
def normalizarDados(dados):
    return dados / 255

# Definição do modelo neural usando PyTorch
class ModeloNeural(nn.Module):
    def __init__(self):
        super(ModeloNeural, self).__init__()
        self.flatten = nn.Flatten()  # Camada para achatar a entrada (ex.: imagem 64x64x3 -> vetor 1D)
        self.fc1 = nn.Linear(64 * 64 * 3, 256)  # Primeira camada densa (256 neurônios)
        self.fc2 = nn.Linear(256, 32)          # Segunda camada densa (32 neurônios)
        self.fc3 = nn.Linear(32, 2)            # Camada de saída (2 neurônios para 2 classes)
        self.tanh = nn.Tanh()                  # Função de ativação Tanh
        self.sigmoid = nn.Sigmoid()            # Função de ativação Sigmoid

    # Função para definir o fluxo de dados através das camadas
    def forward(self, x):
        x = self.flatten(x)        # Achata a entrada
        x = self.tanh(self.fc1(x)) # Passa pela primeira camada e ativa com Tanh
        x = self.tanh(self.fc2(x)) # Passa pela segunda camada e ativa com Tanh
        x = self.sigmoid(self.fc3(x)) # Passa pela camada de saída e ativa com Sigmoid
        return x

# Bloco principal
if __name__ == "__main__":
    print("Carregando os conjuntos de dados de treinamento e teste dos arquivos...")
    
    # Carrega e normaliza os dados de treinamento
    trainX, trainY = carregarConjuntoDeDadosDeArquivo("train_catvnoncat.h5")
    trainX = normalizarDados(trainX)
    
    # Carrega e normaliza os dados de teste
    testX, testY = carregarConjuntoDeDadosDeArquivo("test_catvnoncat.h5")
    testX = normalizarDados(testX)

    # Converte os dados em tensores PyTorch e ajusta o formato
    trainX = torch.tensor(trainX, dtype=torch.float32).permute(0, 3, 1, 2)  # Ajusta para formato NCHW
    trainY = torch.tensor(trainY, dtype=torch.long)  # Rótulos como inteiros longos
    testX = torch.tensor(testX, dtype=torch.float32).permute(0, 3, 1, 2)
    testY = torch.tensor(testY, dtype=torch.long)

    # Divide os dados de treinamento para criar um conjunto de validação
    splitTrainX, valX, splitTrainY, valY = train_test_split(trainX, trainY, test_size=0.2)
    
    # Criação de datasets PyTorch para treinamento, validação e teste
    trainDataset = TensorDataset(splitTrainX, splitTrainY)
    valDataset = TensorDataset(valX, valY)
    testDataset = TensorDataset(testX, testY)

    # Carregadores de dados (DataLoaders) para manipulação em lotes
    trainLoader = DataLoader(trainDataset, batch_size=len(trainDataset), shuffle=True)
    valLoader = DataLoader(valDataset, batch_size=len(valDataset))
    testLoader = DataLoader(testDataset, batch_size=len(testDataset))

    # Inicializa o modelo, a função de perda e o otimizador
    model = ModeloNeural()
    criterion = nn.CrossEntropyLoss()  # Função de perda para classificação multi-classe
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Otimizador Adam com taxa de aprendizado de 0.001

    # Número de épocas para treinamento
    num_epochs = 1000

    # Dicionário para armazenar o histórico de treinamento
    history = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}

    # Loop de treinamento
    for epoch in range(num_epochs):
        # Modo de treinamento
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in trainLoader:
            optimizer.zero_grad()           # Zera os gradientes
            outputs = model(inputs)         # Forward: obtém as predições
            loss = criterion(outputs, labels)  # Calcula a perda
            loss.backward()                 # Backward: ajusta os gradientes
            optimizer.step()                # Atualiza os pesos
            train_loss += loss.item()
            _, predicted = outputs.max(1)   # Obtém as predições
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        train_acc = correct / total

        # Modo de validação
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valLoader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total

        # Armazena os resultados no histórico
        history["accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)
        history["loss"].append(train_loss / len(trainLoader))
        history["val_loss"].append(val_loss / len(valLoader))

        if epoch % 10 == 0:
            print(f"Época {epoch+1}/{num_epochs} - Loss: {train_loss:.4f}, Acurácia: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acurácia: {val_acc:.4f}")

   # Avaliação no conjunto de teste
    model.eval()
    train_preds = []
    train_labels = []
    test_preds = []
    test_labels = []

    # Avaliação no conjunto de treinamento completo
    with torch.no_grad():
        for inputs, labels in DataLoader(TensorDataset(trainX, trainY), batch_size=len(trainX)):
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            train_preds.extend(predicted.numpy())
            train_labels.extend(labels.numpy())

    # Avaliação no conjunto de teste
    with torch.no_grad():
        for inputs, labels in testLoader:
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_preds.extend(predicted.numpy())
            test_labels.extend(labels.numpy())

    # Gráficos
    epochs = range(1, num_epochs + 1)
    plot.figure()
    plot.plot(epochs, history["accuracy"], label="Acurácia de Treinamento")
    plot.plot(epochs, history["val_accuracy"], label="Acurácia de Validação")
    plot.title("Acurácia durante o treinamento")
    plot.xlabel("Épocas")
    plot.ylabel("Acurácia")
    plot.legend()

    plot.figure()
    plot.plot(epochs, history["loss"], label="Perda de Treinamento")
    plot.plot(epochs, history["val_loss"], label="Perda de Validação")
    plot.title("Perda durante o treinamento")
    plot.xlabel("Épocas")
    plot.ylabel("Perda")
    plot.legend()

    # Matriz de confusão do conjunto de treinamento completo
    ConfusionMatrixDisplay.from_predictions(
        train_labels, train_preds, display_labels=["Gato", "Não-Gato"], cmap=plot.cm.Blues
    )
    plot.title("Matriz de Confusão no Conjunto de Treinamento Completo")

    # Matriz de confusão do conjunto de teste
    ConfusionMatrixDisplay.from_predictions(
        test_labels, test_preds, display_labels=["Gato", "Não-Gato"], cmap=plot.cm.Blues
    )
    plot.title("Matriz de Confusão no Conjunto de Teste")

    # Cálculo e impressão da acurácia do conjunto de treinamento completo
    acuracia_treinamento = accuracy_score(train_labels, train_preds)
    print(f"Acurácia no Conjunto de Treinamento Completo: {acuracia_treinamento * 100:.2f}%")

    # Cálculo e impressão da acurácia do conjunto de teste
    acuracia_teste = accuracy_score(test_labels, test_preds)
    print(f"Acurácia no Conjunto de Teste: {acuracia_teste * 100:.2f}%")

    plot.show()
