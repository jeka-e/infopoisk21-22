#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json

import nltk
nltk.download("stopwords")

from nltk.corpus import stopwords
from string import punctuation

import pymorphy2

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity


morph = pymorphy2.MorphAnalyzer()
russian_stopwords = stopwords.words("russian")

tfidfvectorizer = TfidfVectorizer()
tfvectorizer = TfidfVectorizer(use_idf=False)
countvectorizer = CountVectorizer()

def extract_data():
    with open('questions_about_love.jsonl', 'r') as f:
        corpus = list(f)[:50000]

    docs = []
    for el in corpus:
        el = json.loads(el)
        value_max = -1
        text = ''
        for answ in el['answers']:
            if answ['author_rating']['value'] == '':
                value_curr = 0
            else:
                value_curr = int(answ['author_rating']['value'])
            if value_curr > value_max:
                text = answ['text']
                value_max = value_curr
        if text != '':
            docs.append(text)
            
    return docs


def preprocess_text(text):
    tokens = [token.strip('\\,/.\'!?-:;\")(1234567890').lower()               for token in text.split()]        #не понимаю, в чем лог ошибка, вроде все ок же работает...
    
    text = [morph.parse(token)[0].normal_form for token in tokens if token != '']
    
    tokens = ' '.join(token for token in text if token not in russian_stopwords)
    return tokens


def index_corpus(vectorizer, docs):
    X = vectorizer.fit_transform(docs)
    return X


def index_request(vectorizer, request):
    request_indexed = vectorizer.transform(request)
    return request_indexed


def sort_documents(similarity_vec, filelist):
    order = np.argsort(-similarity_vec)    
    return filelist[order]


def precount_similarity_matrix(X_cv, X_tf):
    k = 2.0
    b = 0.75

    l_d = X_cv.sum(axis=1)
    avg_l = l_d.mean()

    IDF = tfidfvectorizer.idf_
    TF = X_tf

    values = []
    rows = []
    cols = []

    for doc, word in zip(*TF.nonzero()): 
        nominator = TF[doc, word] * IDF[word] * (k + 1) 
        denominator = TF[doc, word] + k * (1 - b + b * int(l_d[doc]) / avg_l)
        values.append(nominator/denominator)
        rows.append(doc)
        cols.append(word)
    
    similarity_matrix = sparse.csr_matrix((values, (rows, cols)))
    
    return similarity_matrix


def count_similarity(sim_matrix, request_indexed):
    similarity_vec = sim_matrix.dot(request_indexed.T)
    similarity_vec = np.squeeze(similarity_vec.toarray(), axis=1)
    return similarity_vec


def menu(sim_matrix, answers):
    request = input('Для того, чтобы выйти, введите 0. \nВаш запрос:  ')
    if request != '0':
        request_indexed = index_request(countvectorizer, [preprocess_text(request)])
        similarity_vec = count_similarity(sim_matrix, request_indexed)
        if similarity_vec.sum() == 0:
            print('Таких слов нет в ответах')
            answ = []
        else:
            answ = sort_documents(similarity_vec, answers)
            print('Самые подходящия ответы на ваш запрос:  ', '\n', answ[0], '\n', answ[1], '\n', answ[2], '\n')
        menu(sim_matrix, answers)

        
def main():    
    docs = extract_data()
    answers = np.array(docs)
    
    docs = list(map(preprocess_text, docs))
    
    X_tfidf = index_corpus(tfidfvectorizer, docs) 
    X_tf = index_corpus(tfvectorizer, docs) 
    X_cv = index_corpus(countvectorizer, docs)
    
    sim_matrix = precount_similarity_matrix(X_cv, X_tf)

    menu(sim_matrix, answers)
    

    
main()


# In[ ]:




