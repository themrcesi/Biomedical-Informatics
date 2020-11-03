# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 11:51:16 2020

@author: César
"""

from sklearn.linear_model import LogisticRegression
import numpy as np
import math
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def computeRF(query):
    words = query.split(" ")
    rf = {}
    for word in words:
        try:
            rf[word] += 1
        except:
            rf[word] = 1
    return rf       

def loadDataset(pathDataset):
    """
    query, document, isRelevant
    query, document, isRelevant
    ...
    """
    csv = pd.read_csv(pathDataset, delimiter=";", header=0)
    queries = np.unique(csv["query"].to_numpy())
    
    relevants = [csv.loc[csv["query"]==query]["isRelevant"] for query in queries]
    
    queriesRF = [computeRF(query) for query in queries]
    return queries, queriesRF , csv["term"], relevants

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

def predict(query, clf, documents):
    queryRF = computeRF(query)
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
    print("This is the ranking for query \""+query+"\".....................")
    print(documents.loc[order[:3]])
    
    
if __name__ == "__main__":
    #queries, queriesRF, documents, isRelevant = loadDataset(path)

    file1 = open('../q1.txt', 'r') 
    documents = pd.Series(file1.readlines(), name="document").str.lower()
    queriesRF = [{"glucose":1/2, "blood":1/2}, {"bilirubin":1/2, "plasma":1/2}, {"white":1/4, "blood":1/4, "cells":1/4, "count":1/4}]
    queries = ["glucose blood", "bilirubin plasma", "white blood cell counts"]
    isRelevants = [pd.Series([0 for i in range(67)], name='isRelevant'), pd.Series([0 for i in range(67)], name='isRelevant'),pd.Series([0 for i in range(67)], name='isRelevant')]
    
    clf = LogisticRegression(random_state=0)
    logors = []
    
    for i in range(len(queries)):
    
        queryRF = queriesRF[i]
        isRelevant = isRelevants[i]
        
        if i == 0:
            isRelevant.loc[18] = 1
            isRelevant.loc[23] = 1
        elif i==1:
            isRelevant.loc[4] = 1
        elif i==2:
            isRelevant.loc[14] = 1
            isRelevant.loc[22] = 1
        
        pr = isRelevant.sum() / len(isRelevant)
        logor = math.log(pr*(1-pr))
        logors.append(logor)
        
        tfidf_query = compute_tfidf(documents, queryRF)
        rf = pd.DataFrame([[math.log(queryRF[word]) for word in queryRF] for i in range(len(documents))], columns=[word for word in queryRF])
        
        logodds = []
        
        for a in queryRF:
            tfidf_a = tfidf_query[a]
            rf_a = rf[a]
            
            X = pd.concat((rf_a, tfidf_a), axis = 1)
            y = isRelevant
             
             
            clf.fit(X, y)
            logods_a = clf.predict_log_proba(X)[:,1]
            logodds.append(logods_a)
        
        logodds = np.array(logodds).transpose()
        logodds = (logor + (logodds-logor).sum(axis = 1))
        order = np.flip(np.argsort(logodds))
        print("This is the ranking for query \""+queries[i]+"\".....................")
        print(documents.loc[order[:3]])
        
        
        
        
        
        
        
    
