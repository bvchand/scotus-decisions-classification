from sklearn import model_selection, naive_bayes, svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle


def scotus_rf(X, y, val=False):
    if not val:
        classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
        classifier.fit(X, y)
        pickle.dump(classifier,
                    open("/Users/bharathi/PythonWorkspace/scotus_decisions_application/pickles/random_forest.pkl",
                         "wb"))
        return

    classifier = pickle.load(
        open("/Users/bharathi/PythonWorkspace/scotus_decisions_application/pickles/random_forest.pkl", "rb"))
    pred = classifier.predict(X)
    accuracy = accuracy_score(pred, y) * 100
    print("Random Forest Classifier validation Score -> ", accuracy)

    return pred


def scotus_svm(X, y, val=False):
    if not val:
        classifier = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
        classifier.fit(X, y)
        pickle.dump(classifier,
                    open("/Users/bharathi/PythonWorkspace/scotus_decisions_application/pickles/svm.pkl", "wb"))
        return

    classifier = pickle.load(
        open("/Users/bharathi/PythonWorkspace/scotus_decisions_application/pickles/svm.pkl", "rb"))
    pred = classifier.predict(X)
    accuracy = accuracy_score(pred, y) * 100
    print("SVM Classifier validation Score -> ", accuracy)

    return pred


def scotus_naive_bayes(X, y, val=False):
    if not val:
        classifier = naive_bayes.MultinomialNB()
        classifier.fit(X, y)
        pickle.dump(classifier,
                    open("/Users/bharathi/PythonWorkspace/scotus_decisions_application/pickles/nb.pkl", "wb"))
        return

    classifier = pickle.load(
        open("/Users/bharathi/PythonWorkspace/scotus_decisions_application/pickles/nb.pkl", "rb"))
    pred = classifier.predict(X)
    accuracy = accuracy_score(pred, y) * 100
    print("NB Classifier validation Score -> ", accuracy)

    return pred


def scotus_lr(X, y, val=False):
    if not val:
        classifier = LogisticRegression(max_iter=1000)
        classifier.fit(X, y)
        pickle.dump(classifier,
                    open("/Users/bharathi/PythonWorkspace/scotus_decisions_application/pickles/lr.pkl", "wb"))
        return

    classifier = pickle.load(
        open("/Users/bharathi/PythonWorkspace/scotus_decisions_application/pickles/lr.pkl", "rb"))
    pred = classifier.predict(X)
    accuracy = accuracy_score(pred, y) * 100
    print("LR Classifier validation Score -> ", accuracy)

    return pred
