import argparse

from classifier import create_model, predict, loadModel, saveModel
from dataset import load_dataset
import pandas as pd
import time 

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="dataset path")
    parser.add_argument("-m", "--model", help="already created model path", default="")
    parser.add_argument("-q", "--queries", action='append', help="query that you want to map")
    parser.add_argument("-r", "--ranks", help = "number of terms to recommend", type = int, default = 3, choices = [1,2,3,4,5])
    
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    print("###########################################################")
    print("Starting the ML Ranking based on staged logistic regression")
    print("-----------------------------------------------------------")
    time.sleep(1)

    args = load_args()
       
    queries = args.queries if args.queries is not None else []
    dataset_path = args.dataset
    path_load = args.model
        
    if path_load == "":
        model, logor = create_model(dataset_path)
        print("Model created...")
        # Ask the user to save the model
        isSaved = input("Do you want to save the classifier? ")
        if isSaved == "y":
            saveModel(model, logor)
        else:
            print("Why don´t you like the classifier= :(")
    else:
        model, logor = loadModel(path_load)
        print("Model loaded...")
    time.sleep(1)
    if len(queries) > 0:
        print("-------------------------------------------------------")
        print("Starting the predictions...")
        print("-------------------------------------------------------")
        time.sleep(2)
        _,_,documents,_ = load_dataset(dataset_path)
        
        for query in queries:
            predicted = predict(query, model, documents, args.ranks, logor) 
            print("This is the ranking for query \""+query+"\".....................")
            print(predicted)
            print("-------------------------------------------------------")
        print("Predictions finished...")
        print("#######################################################")
    
    
