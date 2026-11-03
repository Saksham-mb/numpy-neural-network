from data import DataExtracter
import model
from numpy.random import seed
from pathlib import Path
from matplotlib import pyplot as plt
import pickle
import json

seed(42)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / 'models'
FIGURES_DIR=PROJECT_ROOT / 'results' / 'figures'

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

train_data_loader=DataExtracter()
X_train, y_train = train_data_loader.train_data()
model_0=model.NumpyNeuralNetwork(X_train.shape[1], 1, 64, 64)

epochs=2000
lr=0.01

losses=[]

for epoch in range(epochs):
    preds=model_0.forward(X_train)
    loss=model_0.compute_loss(preds)
    losses.append(loss)
    model_0.backward(X_train, y_train)
    model_0.optimize(lr=lr)
    labels=(preds>0.50).astype(float)

acc=(labels==y_train).mean()
print(f'Final Training Loss: {losses[-1]:.4f}')
print(f'Training accuracy: {acc*100:.2f}%')

hyperparameters = {'n_x': model_0.n_x,'n_y': model_0.n_y,'n_h_1': model_0.n_h_1,'n_h_2': model_0.n_h_2,'epochs': epochs,'lr': lr}

with open(MODEL_DIR / 'config.json', 'w') as f:
    json.dump(hyperparameters, f, indent=4)

with open(MODEL_DIR / 'NumpyNeuralNetwork.pkl', 'wb') as f:
    pickle.dump(model_0.params, f)

plt.figure(figsize=(10,7))
plt.title('Numpy Neural Network training loss curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True, alpha=0.7)
plt.plot(range(epochs), losses, c='b')
plt.savefig(FIGURES_DIR / 'loss_curve.png', dpi=300, bbox_inches='tight')
plt.close()