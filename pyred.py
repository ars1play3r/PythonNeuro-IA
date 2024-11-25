import numpy as np

class Capa:
    def __init__(self, n_entradas, n_neuronas):
        self.W = np.random.rand(n_entradas, n_neuronas)
        self.b = np.random.rand(n_neuronas)
    
    def funcion_act(self, z):
        return 1 / (1 + np.exp(-z))  # Sigmoide

class RedNeuronal:
    def __init__(self, neuronas):
        self.capas = []
        for i in range(len(neuronas) - 1):
            self.capas.append(Capa(neuronas[i], neuronas[i + 1]))
    
    def predecir(self, X):
        for capa in self.capas:
            z = X @ capa.W + capa.b
            X = capa.funcion_act(z)
        return X

# Ejemplo de uso
neuronas = [2, 4, 8, 1]  # Estructura: entrada (2), ocultas (4 y 8), salida (1)
red_neuronal = RedNeuronal(neuronas)

# Datos de entrada
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predicciones = red_neuronal.predecir(X)
print(predicciones)
