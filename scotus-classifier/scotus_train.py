from sklearn.model_selection import train_test_split
from scotus_data_preprocess import text_to_doc2vec
from scotus_data_extraction import extract_data
from sklearn import preprocessing
import numpy as np
import pandas as pd
from os import path
import pickle


def data_extraction():
    data_path = "/Users/bharathi/PythonWorkspace/scotus_decisions_application/data/scotus_sample.csv"
    if not path.exists(data_path):
        scotus_dataset, scotus_sample = extract_data()

    df = pd.read_csv(data_path)

    X = df['scotus_text'].to_list()
    y = df['scotus_issue_area'].to_list()

    return X, y


def standardize_split(X, y, val=False):
    length = len(X)
    # print(X)
    d2v_model = text_to_doc2vec(X)
    # print(X)
    X = np.array([d2v_model.docvecs[str(i)] for i in range(length)])
    y = np.array(y)

    scaler = preprocessing.MinMaxScaler()
    X = scaler.fit_transform(X)
    pickle.dump(scaler,
                open("pickles/std_scaler.pkl", "wb"))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
    return X_train, X_test, y_train, y_test

