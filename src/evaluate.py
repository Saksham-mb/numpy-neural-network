import pickle
from pathlib import Path
import model
import json
from data import DataExtracter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / 'models'

with open(MODEL_DIR / 'NumpyNeuralNetwork.pkl', 'rb') as f:
    trained_params=pickle.load(f)

with open(MODEL_DIR / 'config.json', 'r') as f:
    hp = json.load(f)

data_loader=DataExtracter()
X_test, y_test = data_loader.test_data()

model_0 = model.NumpyNeuralNetwork(n_x=hp['n_x'],n_y=hp['n_y'],n_h_1=hp['n_h_1'],n_h_2=hp['n_h_2'])
model_0.params = trained_params

model_0=model.NumpyNeuralNetwork(X_test.shape[1], 1, 64, 64)

model_0.params=trained_params

preds=model_0.forward(X_test)

labels=(preds>0.50).astype(float)

acc=(labels == y_test).mean().astype(float)

print(f'Test accuracy: {acc*100:.2f}%')