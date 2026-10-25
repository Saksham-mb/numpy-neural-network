import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path

MAIN_PATH = Path(__file__).resolve().parent.parent


class data_extracter:
    def __init__(self, MAIN_PATH : Path = MAIN_PATH):
        self.df = pd.read_csv(MAIN_PATH / 'datasets' / 'Titanic-data' / 'train.csv')
        self.df['FamilySize'] = self.df['SibSp'] + self.df['Parch'] + 1
        self.df = self.df.dropna(subset=['Embarked'])
        self.df = pd.get_dummies(self.df, columns=['Sex', 'Embarked'], drop_first=True)

        self.X = self.df.drop(columns=['PassengerId', 'Survived', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin'])
        self.y = self.df['Survived']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, stratify=self.y, random_state=42, test_size=0.2)

    def train_data(self):
        self.X_train['Age'] = self.X_train['Age'].fillna(self.X_train['Age'].median())
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        y_train_mat = self.y_train.values.reshape(-1, 1)

        return X_train_scaled, y_train_mat

    def test_data(self):
        self.X_test['Age'] = self.X_test['Age'].fillna(self.X_train['Age'].median())
        scaler = StandardScaler()
        X_test_scaled = scaler.transform(self.X_test)
        y_test_mat = self.y_test.values.reshape(-1, 1)

        return X_test_scaled, y_test_mat