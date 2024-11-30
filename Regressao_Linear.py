import numpy as np
import h5py
import matplotlib.pyplot as plot
import sklearn.metrics

# Função para carregar o conjunto de dados de um arquivo HDF5
def carregarConjuntoDeDadosDeArquivo(nome_arquivo):
    data = h5py.File(nome_arquivo, 'r')
    # Verifica se é um conjunto de treinamento
    if 'train_set_x' in data:
        dataX = np.array(data['train_set_x'][:])  # Dados de entrada
        dataY = np.array(data['train_set_y'][:])  # Rótulos de saída
    # Verifica se é um conjunto de teste
    elif 'test_set_x' in data:
        dataX = np.array(data['test_set_x'][:])  # Dados de entrada
        dataY = np.array(data['test_set_y'][:])  # Rótulos de saída
    else:
        print("Arquivo inesperado\nEncerrando...")
        quit()
    # Classes presentes no conjunto de dados (ex.: gato e não-gato)
    classes = np.array(data['list_classes'][:])
    
    # Ajusta a forma do vetor de rótulos
    dataY = dataY.reshape((1, dataY.shape[0]))
    return dataX, dataY, classes

# Função de ativação sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Função para achatar e normalizar os dados
# Achata as imagens para vetores unidimensionais e normaliza os valores para o intervalo [0, 1]
def achatarENormalizarDados(dados):
    return dados.reshape(dados.shape[0], -1).T / 255

# Função para calcular a perda (loss) de entropia cruzada
def calcularPerda(Y, m, sigmoid):
    return -(1 / (2 * m)) * np.sum(np.multiply(Y, np.log(sigmoid)) + np.multiply(1 - Y, np.log(1 - sigmoid)))

# Função para implementar o gradiente descendente
# Atualiza os pesos e o viés com base no gradiente e na taxa de aprendizado
def gradienteDescendente(X, Y, matrizPesos, viés, taxaAprendizado, numIteracoes):
    perdasTotais = []
    for it in range(numIteracoes):
        # Propagação direta para calcular o valor sigmoide
        valorSigmoide = sigmoid(np.dot(matrizPesos.T, X) + viés)
        
        # Calcula a perda
        perda = calcularPerda(Y, X.shape[1], valorSigmoide)
        perdasTotais.append(perda)

        # Propagação reversa para calcular os gradientes e atualizar os parâmetros
        gradienteViés = (1 / X.shape[1]) * np.sum(valorSigmoide - Y)
        viés = viés - taxaAprendizado * gradienteViés

        gradienteMatrizPesos = (1 / X.shape[1]) * np.dot(X, (valorSigmoide - Y).T)
        matrizPesos = matrizPesos - taxaAprendizado * gradienteMatrizPesos
    
    return matrizPesos, viés, perdasTotais

# Função para treinar o modelo
def treinarModelo(X, Y, taxaAprendizado, numIteracoes):
    # Inicializa a matriz de pesos com zeros e o viés com 0
    viés = 0
    matrizPesos = np.zeros((X.shape[0], 1))
    
    # Ajusta os pesos e o viés com base no gradiente descendente
    matrizPesos, viés, perdas = gradienteDescendente(X, Y, matrizPesos, viés, taxaAprendizado, numIteracoes)
    return matrizPesos, viés, perdas

# Função para prever se a imagem é de um gato ou não
def preverValores(X, Y, matrizPesos, viés, rotulo):
    previsaoY = np.zeros((1, X.shape[1]))
    matrizPesos = matrizPesos.reshape(X.shape[0], 1)
    valorSigmoide = sigmoid(np.dot(matrizPesos.T, X) + viés)
    
    # Converte as probabilidades em previsões binárias
    for i in range(valorSigmoide.shape[1]):
        previsaoY[0, i] = 1 if valorSigmoide[0, i] > 0.5 else 0
            
    # Calcula e exibe a acurácia
    print(f"Acurácia {rotulo}: {100 - np.mean(np.abs(previsaoY - Y)) * 100} %")
    
    return previsaoY

# Bloco principal
if __name__ == "__main__":
    print("Carregando os conjuntos de dados de treinamento e teste dos arquivos...")
    # Carrega e normaliza os dados de treinamento
    trainX, trainY, trainClasses = carregarConjuntoDeDadosDeArquivo("train_catvnoncat.h5")
    trainX = achatarENormalizarDados(trainX)

    # Carrega e normaliza os dados de teste
    testX, testY, testClasses = carregarConjuntoDeDadosDeArquivo("test_catvnoncat.h5")
    testX = achatarENormalizarDados(testX)

    print("Iniciando o treinamento do modelo...")
    taxaAprendizado = 0.001
    numIteracoes = 5000
    print("Parâmetros: \n- Taxa de Aprendizado: ", taxaAprendizado, "\n- Número de Iterações: ", numIteracoes)
    
    # Treina o modelo
    matrizPesos, viés, perdas = treinarModelo(trainX, trainY, taxaAprendizado, numIteracoes)
        
    print("Fazendo previsões...")
    # Previsões no conjunto de treinamento
    treinamento = preverValores(trainX, trainY, matrizPesos, viés, rotulo="Treinamento")
    # Previsões no conjunto de teste
    previsoesTeste = preverValores(testX, testY, matrizPesos, viés, rotulo="Teste")
        
    # Plota a curva de perda ao longo das iterações
    plot.plot(perdas)
    plot.xlabel("Iterações")
    plot.ylabel("Perda")
    plot.title("Perda ao longo das iterações")
    
    # Calcula e exibe a matriz de confusão para o conjunto de treinamento
    matrizConfusaoTreinamento = sklearn.metrics.confusion_matrix(trainY[0], treinamento[0])
    sklearn.metrics.ConfusionMatrixDisplay(matrizConfusaoTreinamento, display_labels=trainClasses).plot(include_values=True, cmap=plot.cm.Blues, xticks_rotation='horizontal')

    # Calcula e exibe a matriz de confusão para o conjunto de teste
    matrizConfusaoTeste = sklearn.metrics.confusion_matrix(testY[0], previsoesTeste[0])
    sklearn.metrics.ConfusionMatrixDisplay(matrizConfusaoTeste, display_labels=testClasses).plot(include_values=True, cmap=plot.cm.Blues, xticks_rotation='horizontal')
    
    plot.show()
