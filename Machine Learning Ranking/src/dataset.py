# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 23:26:22 2020

@author: César
"""
import pandas as pd
import numpy as np

def compute_rf(query):
    words = query.split(" ")
    rf = {}
    length = len(words)
    for word in words:
        try:
            rf[word] += 1/length
        except:
            rf[word] = 1/length
    return rf 

def load_dataset(pathDataset):
    """
    query, document, isRelevant
    query, document, isRelevant
    ...
    """
    csv = pd.read_csv(pathDataset, delimiter=",", header=0)
    queries = np.unique(csv["query"].to_numpy())
    documents =  csv["term"].iloc[:67]
    relevants = [csv.loc[csv["query"]==query]["isRelevant"] for query in queries]
    
    queriesRF = [compute_rf(query) for query in queries]
    return queries, queriesRF , documents, relevants