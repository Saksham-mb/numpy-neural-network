import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path

MAIN_PATH = Path(__file__).resolve().parent()


def data_extracter(MAIN_PATH : Path = MAIN_PATH):
    df = pd.read_csv(MAIN_PATH / 'datasets' / 'Titanic-data' / 'train.csv')
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df = df.dropna(subset=['Embarked'])
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

    X = df.drop(columns=['PassengerId', 'Survived', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin'])
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)

    X_train['Age'] = X_train['Age'].fillna(X_train['Age'].median())
    X_test['Age'] = X_test['Age'].fillna(X_train['Age'].median())

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_train_mat = y_train.values.reshape(-1, 1)
    y_test_mat = y_test.values.reshape(-1, 1)

    return X_train_scaled, X_test_scaled, y_train_mat, y_test_mat