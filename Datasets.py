import numpy as np
from sklearn.datasets import load_iris, fetch_openml, load_wine
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def get_iris_dataset():
    data = load_iris()
    X_full = data.data
    y_full = np.array([data.target_names[y]
                      for y in data.target.reshape(-1, 1)])
    return X_full, y_full


def get_penguins_dataset():
    df, tgt = fetch_openml(name="penguins", return_X_y=True,
                           as_frame=True, parser='auto')

    df.drop(columns=["island", "sex"], inplace=True)

    mask = df.isna().sum(axis=1) == 0
    df = df[mask]
    tgt = tgt[mask]

    return df.values, tgt.to_numpy().reshape(-1, 1)


def get_wine_dataset():
    data = load_wine()
    X_full = data.data
    y_full = np.array([data.target_names[y]
                      for y in data.target.reshape(-1, 1)])
    return X_full, y_full


def get_letters_dataset():
    letter = fetch_openml('letter', version=1, as_frame=False)
    return letter.data, letter.target.reshape(-1, 1)


def label_encode(y_full):
    return LabelEncoder().fit_transform(y_full.flatten()).reshape(y_full.shape)


def split_transpose(X, y, test_size, random_state):
    return [elem.T for elem in train_test_split(X, y, test_size=test_size, random_state=random_state)]
