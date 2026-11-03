import pickle
from pathlib import Path
import model
import json
from data import DataExtracter
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / 'models'
RESULT_DIR= PROJECT_ROOT / 'results'

with open(MODEL_DIR / 'NumpyNeuralNetwork.pkl', 'rb') as f:
    trained_params=pickle.load(f)

with open(MODEL_DIR / 'config.json', 'r') as f:
    hp = json.load(f)

data_loader=DataExtracter()
X_test, y_test = data_loader.test_data()
X_train, y_train =data_loader.train_data()

model_0 = model.NumpyNeuralNetwork(n_x=hp['n_x'],n_y=hp['n_y'],n_h_1=hp['n_h_1'],n_h_2=hp['n_h_2'])
model_0.params = trained_params

model_0_preds=model_0.forward(X_test)
model_0_labels=(model_0_preds>0.50).astype(float)
model_0_acc=(model_0_labels == y_test).mean().astype(float)

model_1=LogisticRegression(random_state=42)
model_1.fit(X_train, y_train)
model_1_acc=float(model_1.score(X_test, y_test))

metrics={
    'numpy_nn':{
        'test_accuracy':round(model_0_acc * 100, 2)
    },
    'logistic_regression':{
        'test_accuracy':round(model_1_acc*100, 2)
    }
}

with open(RESULT_DIR / 'metrics.json', 'w') as f:
  json.dump(metrics, f, indent=4)

print(f'Custom NumPy NN Test Accuracy: {model_0_acc * 100:.2f}%')
print(f'Logistic Regression Baseline Accuracy: {model_1_acc * 100:.2f}%')