import pickle
from pathlib import Path
import model
from data import data_extracter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / 'models'

with open(MODEL_DIR / 'NumpyNeuralNetwork.pkl', 'rb') as f:
    trained_params=pickle.load(f)

data_loader=data_extracter()
X_test, y_test = data_loader.test_data()

model_0=model.NumpyNeuralNetwork(X_test.shape[1], 1, 64, 64)

model_0.params=trained_params

preds=model_0.forward(X_test)

labels=(preds>0.50).astype(float)

acc=(labels == y_test).mean().astype(float)

print(f'Test accuracy: {acc*100:.2f}%')