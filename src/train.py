from data import data_extracter
import model
from numpy.random import seed as seed
from pathlib import Path
import pickle

seed(42)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / 'models'

MODEL_DIR.mkdir(parents=True, exist_ok=True)

train_data_loader=data_extracter()
X_train, y_train = train_data_loader.train_data()
model_0=model.NumpyNeuralNetwork(X_train.shape[1], 1, 64, 64)

epochs=500
lr=0.01

for epoch in range(epochs):
    preds=model_0.forward(X_train)
    model_0.backward(X_train, y_train)
    model_0.optimize(lr=lr)
    labels=(preds>0.50).astype(float)

acc=(labels==y_train).mean()
print(f'Training accuracy: {acc*100:.2f}%')

with open(MODEL_DIR / 'NumpyNeuralNetwork.pkl', 'wb') as f:
    pickle.dump(model_0.params, f)