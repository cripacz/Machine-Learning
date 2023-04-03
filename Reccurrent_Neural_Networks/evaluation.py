import time
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, plot_confusion_matrix, plot_roc_curve, roc_curve


def GetModelScores (estimator, X_train, X_test, y_train, y_test):

    start = time.time()
    scores = pd.DataFrame(columns= ['Accuracy','F1 Score','Precision','Recall','ROC_AUC'])
    
    model = estimator
    model.fit(X_train, y_train)
    
    prediction_test = model.predict(X_test)
    
        
    try:
        score_test = model.predict_proba(X_test)[:,1]
        roc_test= roc_auc_score(y_test, score_test, average = "weighted")
    except:
        roc_test = 0
    
    scores['Accuracy'] = accuracy_score(y_test, prediction_test),
    scores['F1 Score'] = f1_score(y_test, prediction_test, average = "weighted"),
    scores['Precision'] = precision_score(y_test, prediction_test, average = "weighted"),
    scores['Recall'] = recall_score(y_test, prediction_test, average = "weighted"),
    scores['ROC_AUC'] = roc_test
    end = time.time()

    print(scores)
    print("Inference time: ", end-start)

def GetScoresNN(estimator, X_train, X_test, y_train, y_test, multi=True):

    start =time.time()
    scores = pd.DataFrame(columns= ['Accuracy','F1 Score','Precision','Recall','ROC_AUC'])
    
    model = estimator

    prediction_ts = model.predict(X_test)
    
    if(multi):
      prediction_test = [np.argmax(x) for x in prediction_ts]
    else:
      prediction_test = [round(x[0]) for x in prediction_ts]
        
    try:
        score_test = model.predict_proba(X_test)[:,1]
        roc_test= roc_auc_score(y_test, score_test, average = "weighted")
    except:
        roc_test = 0
           
    scores['Accuracy'] = accuracy_score(y_test, prediction_test),
    scores['F1 Score'] = f1_score(y_test, prediction_test, average = "weighted"),
    scores['Precision'] = precision_score(y_test, prediction_test, average = "weighted"),
    scores['Recall'] = recall_score(y_test, prediction_test, average = "weighted"),
    scores['ROC_AUC'] = roc_test
    end = time.time()

    print(scores)
    print("Inference time: ", end-start)

def GetScores (estimator, X_train, X_test, y_train, y_test):

    start =time.time()
    scores = pd.DataFrame(columns= ['Accuracy','F1 Score','Precision','Recall','ROC_AUC'])
    
    model = estimator

    prediction_test = model.predict(X_test)

    try:
        score_test = model.predict_proba(X_test)[:,1]
        roc_test= roc_auc_score(y_test, score_test, average = "weighted")
    except:
        roc_test = 0
    
               
    scores['Accuracy'] = accuracy_score(y_test, prediction_test),
    scores['F1 Score'] = f1_score(y_test, prediction_test, average = "weighted"),
    scores['Precision'] = precision_score(y_test, prediction_test, average = "weighted"),
    scores['Recall'] = recall_score(y_test, prediction_test, average = "weighted"),
    scores['ROC_AUC'] = roc_test
    end = time.time()
    
    print(scores)
    print("Inference time: ", end-start)