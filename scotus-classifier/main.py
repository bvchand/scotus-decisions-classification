from scotus_models import scotus_rf, scotus_svm, scotus_lr, scotus_naive_bayes
from scotus_train import data_extraction, standardize_split
from scotus_data_preprocess import preprocess_doc2vec
import pickle


if __name__ == "__main__":


    X, y = data_extraction()
    X_train, X_test, y_train, y_test = standardize_split(X, y)

    pred = scotus_svm(X_train, y_train)
    pred = scotus_svm(X_test, y_test, val=True)

    print("... API test")
    nb_classifier = pickle.load(
        open('/Users/bharathi/PythonWorkspace/scotus_decisions_application/pickles/nb.pkl', 'rb'))
    X = preprocess_doc2vec("abc")
    print(X)
    pred = nb_classifier.predict(X)
    print(pred)




