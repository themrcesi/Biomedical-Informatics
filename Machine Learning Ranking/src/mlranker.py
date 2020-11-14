# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 14:26:56 2020

@author: César
"""

import PySimpleGUI as sg
from classifier import loadModel, predict
from dataset import load_dataset
import pandas as pd

query_column = [[sg.Image(filename="../img/loinc.png", pad = ((250,225), (10,10)))],[sg.Text("Please, enter your query:"), sg.In(size=(50,1), enable_events = True, key = "-QUERY-"), sg.Button("Predict", size= (25,1))], [sg.Text("Number of terms to show:"), sg.In(size=(2,1), enable_events = True, key = "-RANKS-")], [sg.Listbox(values = [], enable_events = True, size = (720,50), key ="-PREDICTIONS-")]]
layout = [[sg.Column(query_column)]]

window = sg.Window("Machine Learning Ranking: Staged Logistic Regression", layout, size = (720,480), modal=True)

dataset = None
while(dataset is None):
    dataset = sg.popup_get_file("Please, enter the path of your dataset")

model = None
while(model is None):
    model = sg.popup_get_file("Please, enter the path of your model")
 
logor = None
while(logor is None):
    logor = sg.popup_get_file("Please, enter the path of your logor")
    
model, logor = loadModel(model, logor)
_,_,documents,_ = load_dataset(dataset)

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break
    if event == "Predict":
        query = values["-QUERY-"]
        predicted = predict(query, model, documents, int(values["-RANKS-"]), logor) 
        window["-PREDICTIONS-"].update(predicted)
        
window.close()