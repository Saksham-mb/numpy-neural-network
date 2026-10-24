import numpy as np

class NumpyNeuralNetwork:
    def __init__(self, n_x: int, n_y: int, n_h_1: int = 32, n_h_2: int = 32):
        self.n_x=n_x
        self.n_h_1=n_h_1
        self.n_h_2=n_h_2
        self.n_y=n_y

        self.W1=np.random.randn(n_x, n_h_1)*(np.sqrt(2/n_x))
        self.b1=np.zeros(1, n_h_1)
        self.W2=np.random.randn(n_h_1, n_h_2)*(np.sqrt(2/n_h_1))
        self.b2=np.zeros(1, n_h_2)
        self.W3=np.random.randn(n_h_2, n_y)*(np.sqrt(2/n_h_2))
        self.b3=np.zeros(1, n_y)
    def forward(self, X):
        self.Z1=np.dot(X, self.W1)+self.b1
        self.A1=np.maximum(0, self.Z1)
        self.Z2=np.dot(self.A1, self.W2)+self.b2
        self.A2=np.maximum(0, self.Z2)
        self.Z3=np.dot(self.A2, self.W3)+self.b3
        self.A3=(1/(1+np.exp(-self.Z3)))

        return self.A3
    def backward(self, X, y):
        m=X.shape[0]

        dZ3=self.A3-y
        dW3=(1/m)*np.dot(self.A2.T, dZ3)
        db3=(1/m)*np.sum(dZ3)

        dA2=np.dot(dZ3, self.W3.T)
        dZ2=dA2 * (self.Z2 > 0).astype(float)
        dW2=(1/m)*np.dot(self.A1.T, dZ2)
        db2=(1/m)*np.sum(dZ2, axis=0, keepdims=True)

        dA1=np.dot(dZ2, self.W2.T)
        dZ1=dA1 * (self.Z2 > 0).astype(float)
        dW1=(1/m)*np.dot(self.X.T, dZ1)
        db1=(1/m)*np.sum(dZ1, axis=0, keepdims=True)

        return dW1, db1, dW2, db2, dW3, db3

    def optimizer(self, dW1, db1, dW2, db2, dW3, db3, lr):
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W3 -= lr * dW3
        self.b3 -= lr * db3