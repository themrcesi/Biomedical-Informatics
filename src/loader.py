import json
import xmltodict as xtd
import numpy as np
import os
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import PorterStemmer
from joblib import Parallel, delayed

def preprocess_document(doc, stopset):
    """
    Receives a single document and return a dictionary containing the title and a list of all the stems.

    Parameters
    ----------
    doc : a single document in json.
    stopset : stopset for english

    Returns
    -------
    dict : dictionary having title and stems

    """
    stemmer = PorterStemmer()
    title = np.array([doc["metadata"]["title"]], dtype=str)
    abstract = np.array([paragraph["text"] for paragraph in doc["abstract"]], dtype=str)
    text = np.array([paragraph["text"] for paragraph in doc["body_text"]], dtype=str)
    information = np.concatenate((title, abstract, text))
    tokens = np.concatenate((np.array([wordpunct_tokenize(inf) for inf in information], dtype='object')))
    clean = [token.lower() for token in np.unique(tokens) if token.lower() not in stopset and len(token) > 2]
    final = np.array([stemmer.stem(word) for word in clean])
    return {"title": title[0], "stems": final}

def load_queries(queries_path):
    """
    Receives the path of the queries files and returns a dictionary containing all the queries.

    Parameters
    ----------
    queries_path : path of the queries file

    Returns
    -------
    dic_judgements : dictionary

    """
    with open(queries_path, "r") as xml_file:
        data_dict = xtd.parse(xml_file.read())
    xml_file.close()

    dic_queries = {}
    for query in data_dict["topics"]["topic"]:
        dic_queries[query["@number"]] = query["query"]

    return dic_queries

def load_corpus_parallel(corpus_path):
    """
    Main function to load the documents.

    Parameters
    ----------
    corpus_path : path of the corpus directory

    Returns
    -------
    dic_files : dictionary having a unique entry {paper_id: {title: x, stems: [y,z,w...]}, ...} for each document

    """
    directory = os.fsencode(corpus_path)
    stopset = set(stopwords.words("english"))
    docs = np.array(Parallel(n_jobs=16)(delayed(load_document)(corpus_path, file, stopset) for file in os.listdir(directory)))
    dic_files = {}
    for doc in docs:
        dic_files[doc[0]] = doc[1]
    return dic_files

def load_document(directory_path, file, stopset):
    """
    Specific function to load a single document

    Parameters
    ----------
    directory_path : path of the corpus directory
    file : file to be read
    stopset : stopset for english

    Returns
    -------
    returned : an array having [paper_id, {title: xxxxx, stems: [x,y,z,....]}]

    """
    filename = os.fsdecode(file)
    with open("\\".join([directory_path, filename])) as file:
        file_json = json.load(file)
        returned = np.array([file_json["paper_id"], preprocess_document(file_json, stopset)])
    file.close()
    return returned
      



