import argparse

from classifier import create_model, predict
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
       
    queries = args.queries
    dataset_path = args.dataset
    model = args.model
        
    if model == "":
        model, logor = create_model(dataset_path)
        print("Model created...")
    else:
        model = loadModel(path_load)
        print("Model loaded...")
    time.sleep(1)
    if len(queries) > 0:
        print("-------------------------------------------------------")
        print("Starting the predictions...")
        print("-------------------------------------------------------")
        time.sleep(2)
        #_,_,documents,_ = load_dataset(dataset_path)
        file1 = open(dataset_path, 'r') 
        documents = pd.Series(file1.readlines(), name="document").str.lower()
        for query in queries:
            predict(query, model, documents, args.ranks, logor) 
        print("-------------------------------------------------------")
        print("Predictions finished...")
        print("#######################################################")
    
    
