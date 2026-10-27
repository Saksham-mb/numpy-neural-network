# Titanic Survival Prediction — NumPy Neural Network
A fully custom Neural Network built completely from scratch using numpy matrix calculaion. Solely showcasing neural network basics and what actually happens under the hood in a basic neural network.

## Project Structure

```text
numpy-neural-network/
├── datasets/
│   └── Titanic-Data/
│       └── train.csv
├── models/
│   └── NumpyNeuralNetwork.pkl
├── src/
│   ├── data.py
│   ├── evaluate.py
│   ├── model.py
│   └── train.py
├── .gitignore
├── practice.ipynb
├── README.md
└── requirements.txt

```

(Directory structure based on the project layout).

---

## Model Architecture

The model is a 3-layer deep feedforward artificial neural network implemented in `src/model.py`:

* **Input Layer ($n_x$):** Dynamic number of input features extracted from the preprocessed data.

* **Hidden Layer 1 ($n_{h1}$):** 64 neurons with ReLU activation and He-style weight initialization ($\sqrt{2 / n_x}$).
* 
* **Hidden Layer 2 ($n_{h2}$):** 64 neurons with ReLU activation and He-style weight initialization ($\sqrt{2 / n_{h1}}$).

* **Output Layer ($n_y$):** 1 neuron with Sigmoid activation for binary survival probability prediction.

* **Optimization:** Analytical gradient descent with backpropagation.

* **Decision Threshold:** Predictions greater than 0.50 are classified as positive survival ($1.0$).


### Data Preprocessing:
Features basic feature engineering and EDA involved in this well known Titanic dataset from kaggle.

---

## Setup & Installation

1. **Clone or Download the Repository:**
Ensure the directory structure matches the repository layout.

2. **Prepare Dataset:**
Place `train.csv` inside `datasets/Titanic-Data/`.

3. **Install Dependencies:**
Ensure Python is installed along with the required packages:

```bash
pip install numpy pandas scikit-learn

```


## Usage

Run the scripts from the project root using Python module execution:

### 1. Training the Model

Train the neural network for 500 epochs with a learning rate of 0.01 (`random_state=42`):

```bash
python -m src.train

```

* Fits the network on the training set.

* Outputs final training accuracy.

* Saves the trained weight parameters dictionary to `models/NumpyNeuralNetwork.pkl`.


### 2. Evaluating the Model

Evaluate the saved model parameters against the test partition:

```bash
python -m src.evaluate
```

* Loads `models/NumpyNeuralNetwork.pkl`.

* Extracts the scaled test set features and targets via `data_extracter`.

* Prints the test classification accuracy.