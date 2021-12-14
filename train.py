import pandas as pd
import numpy as np
from nltk.stem.porter import PorterStemmer
import re
import string
from pandas.io import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

def remove_pattern(input_txt,pattern):
    r = re.findall(pattern,input_txt)
    for i in r:
        input_txt = re.sub(i,'',input_txt)

    ##removes puntuactions
    input_txt = re.sub(r'[^\w\s]', ' ', input_txt)

    return input_txt.strip().lower()

if __name__ == '__main__':
    ##loading data
    data = pd.read_csv("sentiment.tsv",sep = '\t')
    data.columns = ["label","body_text"]

    #features and labeles
    data['label'] = data['label'].map({'pos':0,'neg':1})

    data['clean_tweet']=np.vectorize(remove_pattern)(data['body_text'],"@[\w]*")
    tokenized_tweet = data['clean_tweet'].apply(lambda x: x.split())

    stemmer = PorterStemmer()
    tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])

    for i in range(len(tokenized_tweet)):
        tokenized_tweet[i] = ''.join(tokenized_tweet[i])

    data['clean_tweet']=tokenized_tweet

    X = data['clean_tweet']
    y = data['label']

    # Extract Feacture with CountVectorizer
    cv = CountVectorizer()
    X = cv.fit_transform(X)

    clf = LogisticRegression()
    clf.fit(X,y)

    ##save vectorizer and model
    with open('model/logistic_clf.pkl','wb') as f:
        pickle.dump((cv,clf),f)
