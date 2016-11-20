
# coding: utf-8

# In[1]:

import numpy as np
import nltk
import pickle


# In[12]:

def unpickle():
    with open('clf_driver.pkl', 'rb') as fid:
        clf_driver = pickle.load(fid)
    with open('clf_roundtrip.pkl', 'rb') as fid:
        clf_roundtrip = pickle.load(fid)
    with open('clf_relevant.pkl', 'rb') as fid:
        clf_relevant = pickle.load(fid)
    with open('vocab1.pkl', 'rb') as fid:
        vocab1 = pickle.load(fid)
    with open('vocab2.pkl', 'rb') as fid:
        vocab2 = pickle.load(fid)
    return clf_driver, clf_roundtrip, clf_relevant, vocab1, vocab2


# In[ ]:

def feature_vector(msg, vocab):
    stemmer = nltk.stem.porter.PorterStemmer()
    msg = msg.lower()
    tk_text = nltk.word_tokenize(msg)
    stemmed = [stemmer.stem(token) for token in tk_text]
    vec = np.array([[int(voc in stemmed) for voc in vocab]])
    return vec


# In[14]:

def predict(msg, clf_driver, clf_roundtrip, clf_relevant, vocab1, vocab2):
    vec1 = feature_vector(msg, vocab1)
    vec2 = feature_vector(msg, vocab2)
    is_driver = clf_driver.predict(vec1)
    is_roundtrip = clf_roundtrip.predict(vec1)
    is_relevant = clf_relevant.predict(vec2)
    return is_driver, is_roundtrip, is_relevant

clf_driver, clf_roundtrip, clf_relevant, vocab1, vocab2 = unpickle()
for i in predict("I need a ride to new york on friday and a ride back on Tuesday", clf_driver, clf_roundtrip, clf_relevant, vocab1, vocab2):
    print(i)