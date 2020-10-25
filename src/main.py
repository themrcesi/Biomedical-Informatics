import loader as ldr
from gensim import similarities
from nltk.corpus import stopwords

path_queries = r"B:\document_parser\document_parses\topics-rnd5.xml"
path_texts = "B:\document_parser\document_parses\pdf_json"
path_test = "B:/document_parser/document_parses/test"
path_judgements = "B:/document_parser/document_parses/judgements.csv"

queries = ldr.load_queries(path_queries)
model, dictionary, bow, titles_dic = ldr.create_TF_IDF_model(path_test)
judgements = ldr.load_judgements(path_judgements)
#%%
stopset = set(stopwords.words("english"))
index = similarities.MatrixSimilarity(bow, num_features=len(dictionary))
pq = ldr.preprocess_query(queries.iloc[1][0], stopset)
vq = dictionary.doc2bow(pq)
qtfidf = model[vq]
sim = index[qtfidf]
print(sim)


#%%


judgements.loc[judgements['score'] < 1, 'binary_score'] = 0
judgements.loc[judgements['score'] >=1 , 'binary_score'] = 1
