# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 23:42:42 2020

@author: César
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from  sklearn.linear_model import LogisticRegression
import numpy as np
import math
import pandas as pd

from dataset import load_dataset

from joblib import dump, load

def compute_rf(query):
    words = query.split(" ")
    rf = {}
    for word in words:
        try:
            rf[word] += 1
        except:
            rf[word] = 1
    return rf    

def compute_tfidf(documents, query):
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(documents).todense().tolist()
    feature_names = vectorizer.get_feature_names()
    tfidf = pd.DataFrame(vectors, columns=feature_names)
    
    tfidf_query = []
    
    for word in query:
        try:
            tfidf_query.append(pd.DataFrame(tfidf[word], columns=[word]))
        except:
            error = np.zeros((len(documents),1))
            tfidf_query.append(pd.DataFrame(error, columns=[word]))
    if len(tfidf_query) > 1:
        tfidf_query = pd.concat((tfidf_query), axis = 1)
    else:
        tfidf_query = tfidf_query[0]
    return tfidf_query

def predict(query, clf, documents, ranks, logor):
    queryRF = compute_rf(query)
    rf = pd.DataFrame([[math.log(queryRF[word]) for word in queryRF] for i in range(len(documents))], columns=[word for word in queryRF])
    tfidf_query = compute_tfidf(documents, queryRF)
    
    logodds = []
        
    for a in queryRF:
        tfidf_a = tfidf_query[a]
        rf_a = rf[a]
        
        X = pd.concat((rf_a, tfidf_a), axis = 1)
       
        logods_a = clf.predict_log_proba(X)[:,1]
        logodds.append(logods_a)
        
    logodds = np.array(logodds).transpose()
    logodds = (logor + (logodds-logor).sum(axis = 1))
    order = np.flip(np.argsort(logodds))
    return documents.loc[order[:ranks]].values
    
def create_model(model_path):
    queries, queriesRF, documents, isRelevants = load_dataset(model_path)
    
    clf = LogisticRegression()
    logors = []
    X = pd.DataFrame()
    Y = pd.DataFrame()
    
    for i in range(len(queries)):
    
        queryRF = queriesRF[i]
        isRelevant = isRelevants[i]
        
        pr = isRelevant.sum() / len(isRelevant)
        logor = math.log(pr*(1-pr))
        logors.append(logor)
        
        tfidf_query = compute_tfidf(documents, queryRF)
        rf = pd.DataFrame([[math.log(queryRF[word]) for word in queryRF] for i in range(len(documents))], columns=[word for word in queryRF])
        
        logodds = []
        
        for a in queryRF:
            tfidf_a = pd.Series(tfidf_query[a], name="tfidf")
            rf_a = pd.Series(rf[a], name="rf")
            
            x = pd.concat((rf_a, tfidf_a), axis = 1)
            y = isRelevant
             
            X = pd.concat((X,x), axis = 0)
            Y = pd.concat((Y,y), axis = 0)
            
    clf.fit(X, Y)
    return clf, np.sum(logors)/len(logors)

def saveModel(clf, logor):
    dump(clf, 'classifier.joblib') 
    dump(logor, "logor.joblib")
    
def loadModel(model_path, logor_path):
    model = load(model_path)
    logor = load(logor_path)
    return model, logor