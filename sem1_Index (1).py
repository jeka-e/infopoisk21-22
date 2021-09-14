#!/usr/bin/env python
# coding: utf-8

# # ДЗ 1 Индекс

# ## __Задача__: 
# 
# **Data:** Коллекция субтитров сезонов Друзей. Одна серия - один документ.
# 
# **To do:** 
# 
# **1 Создайте обратный индекс этой базы, используя CountVectorizer. 
# То, что вы получите, называется матрица Term-Document.**
# 
# Компоненты вашей реализации:
#     - Функция препроцессинга данных. Включите туда лемматизацию, приведение к одному регистру, удаление пунктуации и стоп-слов.
#     - Функция индексирования данных. На выходе создает обратный индекс, он же матрица Term-Document.
# 
# **2 С помощью обратного индекса посчитайте:** 
# 
# 
# a) какое слово является самым частотным
# 
# b) какое самым редким
# 
# c) какой набор слов есть во всех документах коллекции
# 
# d) кто из главных героев статистически самый популярный (упонимается чаще всего)? Имена героев:
# - Моника
# - Рэйчел 
# - Чендлер
# - Фиби
# - Росс
# - Джоуи, Джои
# 
# **На что направлены эти задачи:** 
# 1. Навык построения обратного индекса
# 2. Навык работы с этим индексом, а именно как с помощью numpy или pandas достать нужную информацию из матрицы данных
# 
# [download_friends_corpus](https://yadi.sk/d/4wmU7R8JL-k_RA?w=1)

# In[176]:


#import zipfile
#with zipfile.ZipFile('./friends-data.zip', 'r') as zip_ref:
#    zip_ref.extractall('./')


# In[177]:


### _check : в коллекции должно быть 165 файлов
import os


filelist = []
for root, dirs, files in os.walk('./friends-data'):
    for name in files:
        filelist.append(os.path.join(root, name))

print(len(filelist))


# In[178]:


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



# In[179]:


docs = [] 

for file in filelist:
    with open(file, 'r') as f:  
        text = f.read() 
        text = preprocess_text(text)   
    docs.append(text)
   


# In[180]:


from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

vectorizer = CountVectorizer(analyzer='word')

def indexing(docs):
    X = vectorizer.fit_transform(docs)
    return X


# In[181]:


X = indexing(docs)
matrix_freq = np.asarray(X.sum(axis=0)).ravel()


# In[182]:


n_max = np.where(matrix_freq == matrix_freq.max())[0][0]
print('1. самое частотное слово: ', vectorizer.get_feature_names()[n_max])


# In[183]:


n_min = np.random.choice(np.where(matrix_freq == matrix_freq.min())[0])
print('2. одно из наименее частотных слов: ', vectorizer.get_feature_names()[n_min])


# In[184]:


matrix_nonzeros = np.count_nonzero(X.toarray(), axis=0)
max_zeros = np.where(matrix_nonzeros == 165)[0]
print('3. набор слов, встречающихся во всех документах: ',      ', '.join(vectorizer.get_feature_names()[max_zero] 
                for max_zero in max_zeros))


# In[189]:


monica = matrix_freq[vectorizer.vocabulary_.get('моника')] + matrix_freq[vectorizer.vocabulary_.get('мон')]

rachel = matrix_freq[vectorizer.vocabulary_.get('рейчел')] + matrix_freq[vectorizer.vocabulary_.get('рейч')]

chandler = matrix_freq[vectorizer.vocabulary_.get('чендлер')] + matrix_freq[vectorizer.vocabulary_.get('чэндлер')] + matrix_freq[vectorizer.vocabulary_.get('чен')]

phoebe = matrix_freq[vectorizer.vocabulary_.get('фиби')] + matrix_freq[vectorizer.vocabulary_.get('фибс')]

ross = matrix_freq[vectorizer.vocabulary_.get('росс')]

joey = matrix_freq[vectorizer.vocabulary_.get('джоуя')] + matrix_freq[vectorizer.vocabulary_.get('джой')] + matrix_freq[vectorizer.vocabulary_.get('джо')]


# In[196]:


print(monica, rachel, chandler, phoebe, ross, joey)


# In[187]:


print('самый популярный персонаж по частоте упоминаний - Росс')


# In[ ]:




