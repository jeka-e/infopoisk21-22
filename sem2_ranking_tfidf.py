#!/usr/bin/env python
# coding: utf-8

# # ДЗ 2  
# ## Ранжирование: TF-IDF, матрица Document-Term, косинусная близость

# ### TfidfVectorizer

# ### __Задача__:    
# 
# Реализуйте поиск, где 
# - в качестве метода векторизации документов корпуса - **TF-IDF**
# - формат хранения индекса - **матрица Document-Term**
# - метрика близости пар (запрос, документ) - **косинусная близость**
# 
# 
# Что должно быть в реализации:
# - функция индексации корпуса, на выходе которой посчитанная матрица Document-Term 
# - функция индексации запроса, на выходе которой посчитанный вектор запроса
# - функция с реализацией подсчета близости запроса и документов корпуса, на выходе которой вектор, i-й элемент которого обозначает близость запроса с i-м документом корпуса
# - главная функция, объединяющая все это вместе; на входе - запрос, на выходе - отсортированные по убыванию имена документов коллекции
# 
# В качестве корпуса возьмите корпус Друзей из первого задания.
# 
# **На что направлена эта задача:** 
# Реализация от начала до конца механики поиска с использованием простых компонентов.
# 

# In[1]:


import os


filelist = []
for root, dirs, files in os.walk('./friends-data'):
    for name in files:
        filelist.append(os.path.join(root, name))


# In[2]:


import nltk
nltk.download("stopwords")

from nltk.corpus import stopwords
from string import punctuation

import pymorphy2


morph = pymorphy2.MorphAnalyzer() 
russian_stopwords = stopwords.words("russian")


def preprocess_text(text):
    tokens = [token.strip('\\,.\'!?-:;\")(1234567890').lower()               for token in text.split()]
    
    text = [morph.parse(token)[0].normal_form for token in tokens]
    
    tokens = ' '.join(token for token in text if token not in russian_stopwords)
    return tokens



# In[3]:


from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import scipy.spatial.distance as ds


vectorizer = TfidfVectorizer()

def index_corpus(docs):
    X = vectorizer.fit_transform(docs)
    return X


def index_request(request):
    request_indexed = vectorizer.transform(request)
    return request_indexed


def count_similarity(X, request):
    return ds.cdist(X.toarray(), request.toarray(), 'cosine').reshape(-1)


def sort_documents(similarity_vec, filelist):
    order = np.argsort(similarity_vec)
    ordered_filelist = []
    for file_number in order:
         ordered_filelist.append(filelist[file_number])
    return ordered_filelist


# In[4]:


import re

docs = [] 
filenames = []
regex = '(\dx\d*.*?)\.ru'

for file in filelist:
    with open(file, 'r') as f:  
        text = f.read() 
        text = preprocess_text(text)   
    docs.append(text)
    name = re.search(regex, file)
    filenames.append(name.group(1))
    
    
X = index_corpus(docs)


# Я бы сделала все через одну функцию, но делаю вторую чисто для того, чтобы была функция, принимающая запрос и возвращающая отсортированный список, как требуется в задании, вот
# этот список я сохраняю в переменную, вывожу на экран только самый подходящий ответ на запрос

# In[16]:


def main(request, X, filenames):
    request_prep = preprocess_text(request)
    request_indexed = index_request([request_prep])
    similarity_vec = count_similarity(X, request_indexed)
    if np.isnan(np.sum(similarity_vec)):
        print('Таких слов нет в словаре')
        answ = []
    else:
        answ = sort_documents(similarity_vec, filenames)
        print('Самый подходящий ответ на ваш запрос:  ', answ[0], '\n')
    return(answ)
    
    


# In[17]:


def menu(X, filenames):
    request = input('Для того, чтобы выйти, введите 0. \nВаш запрос:  ')
    if request != '0':
        answ = main(request, X, filenames)
        menu(X, filenames)
    
    
    


# In[19]:


menu(X, filenames)


# In[ ]:




