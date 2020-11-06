# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 14:26:56 2020

@author: César
"""

import PySimpleGUI as sg
from classifier import loadModel, predict
from dataset import load_dataset
import pandas as pd

query_column = [[sg.Text("Please, enter your query:"), sg.In(size=(25,1), enable_events = True, key = "-QUERY-"), sg.Button("Predict")], [sg.Listbox(values = [], enable_events = True, size = (40,10), key ="-PREDICTIONS-")]]
# query_column = [[sg.Text("Please, enter your the path of your dataset:"), sg.In(size=(25,1), enable_events = True, key = "-DATASET-")], [sg.Text("Please, enter your query:"), sg.In(size=(25,1), enable_events = True, key = "-QUERY-"), sg.Button("Predict")], [sg.Listbox(values = [], enable_events = True, size = (40,10), key ="-PREDICTIONS-")]]
layout = [[sg.Column(query_column)]]

model, logor = loadModel("classifier.joblib")

window = sg.Window("Machine Learning Ranking: Staged Logistic Regression", layout)
dataset = None
while(dataset is None):
    dataset = sg.popup_get_file("Please, enter the path of your dataset")
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break
    if event == "Predict":
        dataset_path = dataset
        query = values["-QUERY-"]
        #_,_,_documents,_ = load_dataset(dataset_path)
        file1 = open(dataset_path, 'r') 
        documents = pd.Series(file1.readlines(), name="document").str.lower()
        predicted = predict(query, model, documents, 3, logor) 
        window["-PREDICTIONS-"].update(predicted)
        
window.close()