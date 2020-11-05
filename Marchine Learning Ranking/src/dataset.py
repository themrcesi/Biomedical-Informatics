# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 23:26:22 2020

@author: César
"""
import pandas as pd
import numpy as np

def load_dataset(pathDataset):
    """
    query, document, isRelevant
    query, document, isRelevant
    ...
    """
    csv = pd.read_csv(pathDataset, delimiter=";", header=0)
    queries = np.unique(csv["query"].to_numpy())
    
    relevants = np.array([csv.loc[csv["query"]==query]["isRelevant"] for query in queries])
    
    queriesRF = [computeRF(query) for query in queries]
    return queries, queriesRF , csv["term"], relevants