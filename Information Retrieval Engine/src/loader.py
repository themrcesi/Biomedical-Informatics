import json
import xmltodict as xtd
import numpy as np
import os
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import PorterStemmer
from joblib import Parallel, delayed
import dask
import dask.array as da
from gensim import corpora
from gensim import models
import pandas as pd
from gensim import similarities


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
    title = np.array([doc["metadata"]["title"]], dtype=str)
    abstract = np.array([paragraph["text"] for paragraph in doc["abstract"]], dtype=str)
    text = np.array([paragraph["text"] for paragraph in doc["body_text"]], dtype=str)
    stemmer = PorterStemmer()
    information = np.concatenate((title, abstract, text))
    tokens = np.concatenate(([wordpunct_tokenize(inf) for inf in information]))
    clean = [token.lower() for token in tokens if token.lower() not in stopset and len(token) > 2]
    final = [stemmer.stem(word) for word in clean]
    return {"title": title[0], "stems": final}

def preprocess_query(query, stopset):
    stemmer = PorterStemmer()
    tokens = wordpunct_tokenize(query)
    clean = [token.lower() for token in tokens if token.lower() not in stopset and len(token) > 2]
    stems = [stemmer.stem(word) for word in clean]
    return stems

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

    df = pd.DataFrame.from_dict(dic_queries, orient='index', columns=['query'])

    return df

def load_corpus_parallel(path_corpus):
    directory = os.fsencode(path_corpus)
    stopset = set(stopwords.words("english"))
    dic = corpora.Dictionary()
    titles_dic = {}
    def load_single_document(file, stopset):
        filename = os.fsdecode(file)
        with open("\\".join([path_corpus, filename])) as file:
            file_json = json.load(file)
        pre_doc = preprocess_document(file_json, stopset)
        titles_dic[file_json["paper_id"]] = pre_doc["title"]
        dic.add_documents([pre_doc["stems"]])
    Parallel(n_jobs=16, backend="threading")(delayed(load_single_document)(file, stopset) for file in os.listdir(directory))   
    return titles_dic, dic

def load_corpus_single(path_corpus):
    directory = os.fsencode(path_corpus)
    stopset = set(stopwords.words("english"))
    dic = corpora.Dictionary()
    titles_dic = {}
    counter = 1        
    for file in os.listdir(directory):
        print(counter)
        counter += 1
        filename = os.fsdecode(file)
        with open("\\".join([path_corpus, filename])) as file:
            file_json = json.load(file)
        pre_doc = preprocess_document(file_json, stopset)
        titles_dic[file_json["paper_id"]] = pre_doc["title"]
        dic.add_documents([pre_doc["stems"]])
        
    df_titles = pd.DataFrame.from_dict(titles_dic, orient = 'index', columns=['title'])
        
    return titles_dic, dic

def doc2bows_parallel(path_corpus, dictionary):
    directory = os.fsencode(path_corpus)
    stopset = set(stopwords.words("english"))
    vectors = []
    def doc2bow_single_document(file):
        filename = os.fsdecode(file)
        with open("\\".join([path_corpus, filename])) as file:
            file_json = json.load(file)
        pre_doc = preprocess_document(file_json, stopset)
        vectors.append(dictionary.doc2bow(pre_doc["stems"]))
    Parallel(n_jobs=16, backend="threading", verbose=100)(delayed(doc2bow_single_document)(file) for file in os.listdir(directory))   
    corpora.MmCorpus.serialize("B:/Proyectos/InformationRetrievalSystem/vsx_docs.mm", vectors)
    return vectors

def doc2bows_single(path_corpus, dictionary):
    directory = os.fsencode(path_corpus)
    stopset = set(stopwords.words("english"))
    vectors = []
    counter = 1
    for file in os.listdir(directory):
        print(counter)
        counter += 1
        filename = os.fsdecode(file)
        with open("\\".join([path_corpus, filename])) as file:
            file_json = json.load(file)
        pre_doc = preprocess_document(file_json, stopset)
        vectors.append(dictionary.doc2bow(pre_doc["stems"]))
        
    corpora.MmCorpus.serialize("B:/Proyectos/InformationRetrievalSystem/vsx_docs.mm", vectors)
    return vectors

def load_judgements(path_judgements):
    judgements = pd.read_csv(path_judgements, delimiter=' ', names = ["query", "document", "score"], usecols=[0,2,3])
    return judgements

def create_TF_IDF_model(path_corpus):
    titles_dic, dictionary = load_corpus_single(path_corpus)
    bow = doc2bows_single(path_corpus, dictionary)
    tfidf = models.TfidfModel(bow)
    return tfidf, dictionary, bow, titles_dic

def launch_query(model, dictionary, bow, query, titles):
    stopset = set(stopwords.words("english"))
    index = similarities.MatrixSimilarity(bow, num_features=len(dictionary))
    pq = preprocess_query(query, stopset)
    vq = dictionary.doc2bow(pq)
    qtfidf = model[vq]
    sim = index[qtfidf]
    print(sim)
