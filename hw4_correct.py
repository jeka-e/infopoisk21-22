import gensim
from gensim.models import KeyedVectors
import numpy as np
import json
import pickle
import tqdm
import nltk
nltk.download("stopwords")

from nltk.corpus import stopwords
from string import punctuation

import pymorphy2

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModel

morph = pymorphy2.MorphAnalyzer()
russian_stopwords = stopwords.words("russian")

tfidfvectorizer = TfidfVectorizer()
tfvectorizer = TfidfVectorizer(use_idf=False)
countvectorizer = CountVectorizer()


fast_text_file = 'araneum_none_fasttextcbow_300_5_2018.model'
model_fasttext = gensim.models.KeyedVectors.load(fast_text_file)

tokenizer_bert = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
model_bert = AutoModel.from_pretrained("cointegrated/rubert-tiny")
model_bert.cuda()  


def extract_data():
    with open('questions_about_love.jsonl', 'r') as f:
        corpus = list(f)[:10000]  

    questions = []
    docs = []
    for el in corpus:
        el = json.loads(el)
        question = el['question']
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
            questions.append(question)

    return docs, questions


def preprocess_text(text):
    tokens = [token.strip('\\,/.\'!?-:;\")(1234567890').lower()  
              for token in text.split()]
    
    text = [morph.parse(token)[0].normal_form for token in tokens if token != '']
    
    tokens = ' '.join(token for token in text if token not in russian_stopwords)
    return tokens


def index_corpus_tfidf_cv(vectorizer, docs):
    X = vectorizer.fit_transform(docs)
    return X


def index_fasttext(docs):
    if type(docs) != list:
        docs = [docs]
    doc_vecs = []
    for doc in docs:
        sum_vec = np.zeros(300)
        sum_n = 0
        if doc == '':
            sum_n = 1
        else: 
            for token in doc.split():
                vec = model_fasttext.get_vector(token)
                sum_vec += vec
                sum_n += 1
        
        doc_vecs.append(sum_vec/sum_n)
    return np.vstack(doc_vecs)


def index_bert(docs):
    doc_vecs = []
    if type(docs) != list:
        docs = [docs] 
    for doc in docs:
        t = tokenizer_bert(doc, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = model_bert(**{k: v.to(model_bert.device) for k, v in t.items()})
        embeddings = model_output.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings)
        doc_vecs.append(embeddings[0].cpu().numpy())
    return np.vstack(doc_vecs)


def index_corpus(vectorizer, docs):
    if vectorizer == 'tfidf':
        X = index_corpus_tfidf_cv(tfidfvectorizer, docs)
    elif vectorizer == 'cv':
        X = index_corpus_tfidf_cv(countvectorizer, docs)
    elif vectorizer == 'bm25':
        X_tfidf = index_corpus_tfidf_cv(tfidfvectorizer, docs) 
        X_tf = index_corpus_tfidf_cv(tfvectorizer, docs) 
        X_cv = index_corpus_tfidf_cv(countvectorizer, docs)
        X = precount_similarity_matrix_bm25(X_cv, X_tf)
    elif vectorizer == 'fasttext':
        X = index_fasttext(docs)
    elif vectorizer == 'bert':
        X = index_bert(docs)
    return X


def index_request_tfidf_cv(vectorizer, request):
    request_indexed = vectorizer.transform(request)
    return request_indexed


def index_request(vectorizer, request):
    if vectorizer == 'tfidf':
        request_indexed = index_request_tfidf_cv(tfidfvectorizer, [preprocess_text(request)])
    elif vectorizer == 'cv':
        request_indexed = index_request_tfidf_cv(countvectorizer, [preprocess_text(request)])
    elif vectorizer == 'bm25':
        request_indexed = index_request_tfidf_cv(countvectorizer, [preprocess_text(request)])
    elif vectorizer == 'fasttext':
        request_indexed = index_fasttext(request)
    elif vectorizer == 'bert':
        request_indexed = index_bert(request)
    return request_indexed


def index_questions(vectorizer, questions):
    if vectorizer == 'cv' or vectorizer == 'bm25':
        questions_indexed = index_request_tfidf_cv(countvectorizer, questions)
    elif vectorizer == 'tfidf':
        questions_indexed = index_request_tfidf_cv(tfidfvectorizer, questions)
    elif vectorizer == 'fasttext':
        questions_indexed = index_fasttext(questions)
    elif vectorizer == 'bert':
        questions_indexed = index_bert(questions)

    return questions_indexed



def count_similarity(sim_matrix, request_indexed):
    similarity_vec = cosine_similarity(sim_matrix, request_indexed)
    return similarity_vec


def count_similarity_bm25(sim_matrix, request_indexed):
    return np.dot(sim_matrix, request_indexed.T).toarray()


def sort_documents(similarity_vec, filelist):
    order = np.argsort(-similarity_vec)[:, :5]
    return filelist[order]


def precount_similarity_matrix_bm25(X_cv, X_tf):
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
        nominator = TF[doc, word]*IDF[word]*(k+1) 
        denominator = TF[doc, word] + k * (1 - b + b * int(l_d[doc]) / avg_l)
        values.append(nominator/denominator)
        rows.append(doc)
        cols.append(word)
    
    similarity_matrix = sparse.csr_matrix((values, (rows, cols)))
    
    return similarity_matrix


def menu(matrixes, answers):
    request = input('Для того, чтобы выйти, введите 0. \nВаш запрос:  ')
    requests = []
    if request != '0':
        for i, vectorizer in enumerate(['tfidf', 'cv', 'bm25', 'fasttext', 'bert']):
            request_indexed = index_request(vectorizer=vectorizer, request=request)
            if vectorizer != 'bm25':
                similarity_vec = count_similarity(matrixes[i], request_indexed)
                similarity_vec = np.squeeze(similarity_vec, axis=1)
            else:
                similarity_vec = count_similarity_bm25(matrixes[i], request_indexed)
                similarity_vec = np.squeeze(similarity_vec, axis=1)
            similarity_vec = np.expand_dims(similarity_vec, axis=0)
            print('\n', vectorizer, ':')
            if similarity_vec.sum() == 0:
                print('Таких слов нет в ответах')
                answ = []
            else:
                answ = sort_documents(similarity_vec, answers)
                answ = np.squeeze(answ, axis=0)
                print('Самые подходящие ответ на ваш запрос:  \n', answ[0], '\n', answ[1], '\n', answ[2], '\n',answ[3], '\n',answ[4])
        menu(matrixes, answers)


def main():
    docs, questions = extract_data()
    answers = docs
    questions_correct = np.array(questions)
    
    docs = list(map(preprocess_text, docs))
    questions = list(map(preprocess_text, questions))

    answers = np.array(answers)
    docs = docs
    questions = questions
    matrixes = []
    for vectorizer in ['tfidf', 'cv', 'bm25', 'fasttext', 'bert']:
        X = index_corpus(vectorizer=vectorizer, docs=docs)
        matrixes.append(X)
        questions_indexed = index_questions(vectorizer=vectorizer, questions=questions)
        similarity_matrix = count_similarity(X, questions_indexed).T  
        docs_sorted = sort_documents(similarity_matrix, np.array(docs))
        score = 0
        for i, doc_sorted in enumerate(docs_sorted):
            if docs[i] in doc_sorted[:5]:
                score += 1
        print(f'{vectorizer}: {score/len(docs)}')
    
    menu(matrixes, answers)

main()

