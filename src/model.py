import numpy as np

class NumpyNeuralNetwork:

    def __init__(self, n_x: int, n_y: int, n_h_1: int = 32, n_h_2: int = 32):
        self.params={}

        self.n_x = n_x
        self.n_h_1 = n_h_1
        self.n_h_2 = n_h_2
        self.n_y = n_y

        self.params['W1'] = np.random.randn(n_x, n_h_1) * np.sqrt(2.0 / n_x)
        self.params['b1'] = np.zeros((1, n_h_1))
        self.params['W2'] = np.random.randn(n_h_1, n_h_2) * np.sqrt(2.0 / n_h_1)
        self.params['b2'] = np.zeros((1, n_h_2))
        self.params['W3'] = np.random.randn(n_h_2, n_y) * np.sqrt(1.0 / n_h_2)
        self.params['b3'] = np.zeros((1, n_y))

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.Z1 = np.dot(X, self.params['W1']) + self.params['b1']
        self.A1 = np.maximum(0, self.Z1)

        self.Z2 = np.dot(self.A1, self.params['W2']) + self.params['b2']
        self.A2 = np.maximum(0, self.Z2)

        self.Z3 = np.dot(self.A2, self.params['W3']) + self.params['b3']
        self.A3 = 1.0 / (1.0 + np.exp(-self.Z3))

        return self.A3

    def backward(self, X: np.ndarray, y: np.ndarray):
        m = X.shape[0]

        self.dZ3 = self.A3 - y
        self.dW3 = (1 / m) * np.dot(self.A2.T, self.dZ3)
        self.db3 = (1 / m) * np.sum(self.dZ3, axis=0, keepdims=True)

        self.dA2 = np.dot(self.dZ3, self.params['W3'].T)
        self.dZ2 = self.dA2 * (self.Z2 > 0).astype(float)
        self.dW2 = (1 / m) * np.dot(self.A1.T, self.dZ2)
        self.db2 = (1 / m) * np.sum(self.dZ2, axis=0, keepdims=True)

        self.dA1 = np.dot(self.dZ2, self.params['W2'].T)
        self.dZ1 = self.dA1 * (self.Z1 > 0).astype(float)
        self.dW1 = (1 / m) * np.dot(X.T, self.dZ1)
        self.db1 = (1 / m) * np.sum(self.dZ1, axis=0, keepdims=True)


    def optimize(self, lr: float = 0.005):
        self.params['W1'] -= lr * self.dW1
        self.params['b1'] -= lr * self.db1
        self.params['W2'] -= lr * self.dW2
        self.params['b2'] -= lr * self.db2
        self.params['W3'] -= lr * self.dW3
        self.params['b3'] -= lr * self.db3