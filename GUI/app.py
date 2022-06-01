import numpy as np
import pandas as pd
import nltk
import copy
import pickle
from sklearn import metrics
from nltk.tag import pos_tag
from sklearn import preprocessing
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import wordnet, stopwords
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from flask import Flask, render_template, url_for, request
from PreProcessing import *

app = Flask(__name__)

@app.route("/" , methods=['POST' , 'GET'])

def test():
    if request.method == "GET":
        return render_template("index.html")
    else:
        Object = Preprocessing()
        Query = request.form['query']
        QueryProcessed1 = Query.lower()
        QueryProcessed2 = Object.removeEncodings(QueryProcessed1)
        QueryProcessed3 = Object.removeStopWords(QueryProcessed2)
        QueryProcessed4 = Object.removePunctuations(QueryProcessed3)
        QueryProcessed5 = Object.lemmatizeSentence(QueryProcessed4)
        query = [QueryProcessed5]

        prediction_All = Predict_Query_All(query)
        prediction_TfIDF = Predicted_Query_TFIDF(query)
        prediction_Topic_Modelling = Predict_Query_Topic_Modelling(query)
        prediction_Lexical_Chain = Predict_Query_Lexical_Chain(query)

        predictions = dict()
        predictions["All Models Combined"] = prediction_All
        predictions["Tf-IDF"] = prediction_TfIDF
        predictions["Topic Modelling"] = prediction_Topic_Modelling
        predictions["Lexical Chain"] = prediction_Lexical_Chain
        return render_template('result.html' , result=predictions , Query=Query)
        
@app.route("/analysis" , methods=['GET']) 
def Model_Analysis():
    if request.method == "GET":
        return 

if __name__ == "__main__":

    def Load_Model_All():
        model = pickle.load(open('Saved_Models/model.pickle' , 'rb'))
        return model

    def Load_Model_Topic_Model():
        model = pickle.load(open('Saved_Models/model_Topic.pickle' , 'rb'))
        return model

    def Load_Model_Lexical():
        model = pickle.load(open('Saved_Models/model_Lexical.pickle' , 'rb'))
        return model
    
    def Load_Model_Tfidf():
        model = pickle.load(open('Saved_Models/model_Tfidf.pickle' , 'rb'))
        return model
        
    def Predicted_Query_TFIDF(query):
        tfidf_vectorizer = pickle.load(open('Saved_Models/vectorizer_Tfidf.pickle' , 'rb'))
        corpus_vocabulary = defaultdict(None, copy.deepcopy(tfidf_vectorizer.vocabulary_))
        corpus_vocabulary.default_factory = corpus_vocabulary.__len__
        model = Load_Model_Tfidf()
        tfidf_transformer_query = TfidfVectorizer()
        tfidf_transformer_query.fit_transform(query)
        for word in tfidf_transformer_query.vocabulary_.keys():
            if word in tfidf_vectorizer.vocabulary_:
                corpus_vocabulary[word]
        
        tfidf_transformer_query_sec = TfidfVectorizer(vocabulary=corpus_vocabulary)
        query_tfidf_matrix = tfidf_transformer_query_sec.fit_transform(query)
        return model.predict(query_tfidf_matrix) 

    def Predict_Query_Topic_Modelling(query):
        tfidf_vectorizer = pickle.load(open('Saved_Models/vectorizer_Topic.pickle' , 'rb'))
        corpus_vocabulary = defaultdict(None, copy.deepcopy(tfidf_vectorizer.vocabulary_))
        corpus_vocabulary.default_factory = corpus_vocabulary.__len__
        model = Load_Model_Topic_Model()
        tfidf_transformer_query = TfidfVectorizer()
        tfidf_transformer_query.fit_transform(query)
        for word in tfidf_transformer_query.vocabulary_.keys():
            if word in tfidf_vectorizer.vocabulary_:
                corpus_vocabulary[word]
        
        tfidf_transformer_query_sec = TfidfVectorizer(vocabulary=corpus_vocabulary)
        query_tfidf_matrix = tfidf_transformer_query_sec.fit_transform(query)
        return model.predict(query_tfidf_matrix) 
    
    def Predict_Query_Lexical_Chain(query):
        tfidf_vectorizer = pickle.load(open('Saved_Models/vectorizer_Lexical.pickle' , 'rb'))
        corpus_vocabulary = defaultdict(None, copy.deepcopy(tfidf_vectorizer.vocabulary_))
        corpus_vocabulary.default_factory = corpus_vocabulary.__len__
        model = Load_Model_Lexical()
        tfidf_transformer_query = TfidfVectorizer()
        tfidf_transformer_query.fit_transform(query)
        for word in tfidf_transformer_query.vocabulary_.keys():
            if word in tfidf_vectorizer.vocabulary_:
                corpus_vocabulary[word]
        
        tfidf_transformer_query_sec = TfidfVectorizer(vocabulary=corpus_vocabulary)
        query_tfidf_matrix = tfidf_transformer_query_sec.fit_transform(query)
        return model.predict(query_tfidf_matrix)

    def Predict_Query_All(query):
        tfidf_vectorizer = pickle.load(open('Saved_Models/vectorizer.pickle' , 'rb'))
        corpus_vocabulary = defaultdict(None, copy.deepcopy(tfidf_vectorizer.vocabulary_))
        corpus_vocabulary.default_factory = corpus_vocabulary.__len__
        model = Load_Model_All()
        tfidf_transformer_query = TfidfVectorizer()
        tfidf_transformer_query.fit_transform(query)
        for word in tfidf_transformer_query.vocabulary_.keys():
            if word in tfidf_vectorizer.vocabulary_:
                corpus_vocabulary[word]
        
        tfidf_transformer_query_sec = TfidfVectorizer(vocabulary=corpus_vocabulary)
        query_tfidf_matrix = tfidf_transformer_query_sec.fit_transform(query)
        return model.predict(query_tfidf_matrix) 

    app.run(debug=True)