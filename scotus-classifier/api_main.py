import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI, Path
import pickle

from scotus_data_preprocess import preprocess_doc2vec

app = FastAPI()

nb_classifier = pickle.load(open('/Users/bharathi/PythonWorkspace/scotus_decisions_application/pickles/nb.pkl', 'rb'))
rf_classifier = pickle.load(
    open('/Users/bharathi/PythonWorkspace/scotus_decisions_application/pickles/random_forest.pkl', 'rb'))
svm_classifier = pickle.load(open('/Users/bharathi/PythonWorkspace/scotus_decisions_application/pickles/svm.pkl', 'rb'))
lr_classifier = pickle.load(open('/Users/bharathi/PythonWorkspace/scotus_decisions_application/pickles/lr.pkl', 'rb'))

models = {"NB": nb_classifier,
          "RF": rf_classifier,
          "SVM": svm_classifier,
          "LR": lr_classifier}


class SCOTUSText(BaseModel):
    model: str
    scotus_text: str


@app.get("/")
def home():
    return {"name": "SCOTUS Decision Classifier"}


@app.get("/get-models")
def get_models():
    return {"models": models.keys()}


@app.post("/predict")
def predict_scotus_decision(input: SCOTUSText):
    input = vars(input)
    print("hi")
    model = input['model']
    if model not in models:
        return {"Error": "Model doesn't exist"}
    X = preprocess_doc2vec(input['scotus_text'])
    pred = models[model].predict(X)
    print(pred[0])
    return pred[0]

