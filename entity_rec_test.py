__author__ = 'Leandra'
import numpy as np
import matplotlib as plt
import pandas as pd
import nltk
import re
import time
import random, math, json
from sutime import SUTime

def load_data(csv_fpath):
        print("loading data")
        data = pd.read_csv(csv_fpath, encoding = 'latin1')

        return data

##let the fun begin!##
def processLanguage(messages):
    vocab = {}
    try:
        for item in messages:
            tokenized = nltk.word_tokenize(item)
            tagged = nltk.pos_tag(tokenized)
            print tagged

            for (word, tag) in tagged:
                if tag == 'NNP':
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
            #namedEnt = nltk.ne_chunk(tagged)
            #namedEnt.draw()


    except Exception, e:
        print str(e)
    return vocab

def processLanguage2(messages):
    vocab = {}
    try:
        for (item, _) in messages:
            tokenized = nltk.word_tokenize(item)
            tagged = nltk.pos_tag(tokenized)
            print tagged

            for (word1, tag1), (word2, tag2) in zip(tagged, tagged[1:]):
                if word1.lower() == 'to' and tag2 == 'NNP':
                    if word2 in vocab:
                        vocab[word2] += 1
                    else:
                        vocab[word2] = 1
            #namedEnt = nltk.ne_chunk(tagged)
            #namedEnt.draw()


    except Exception, e:
        print str(e)
    return vocab

def processLanguage3(messages):
    jar_files = 'C:\Users\Leandra\Documents\Fall2016\NLP\carpool-search\jarsnew'
    print(jar_files)
    sutime = SUTime(jars=jar_files, mark_time_ranges=True)


    try:
        for (item, date) in messages:
            print(item)
            #print(json.dumps(sutime.parsedate(item, date), sort_keys=True, indent=4))
            print(sutime.parsedate(item, date))
            #namedEnt = nltk.ne_chunk(tagged)
            #namedEnt.draw()


    except Exception, e:
        print str(e)


def processLanguage4(messages):
    vocab = {}
    try:
        for (item, _) in messages:
            tokenized = nltk.word_tokenize(item)
            tagged = nltk.pos_tag(tokenized)
            namedEnt = nltk.ne_chunk(tagged)
            print(namedEnt)
            #namedEnt.draw()


    except Exception, e:
        print str(e)
    return vocab

if __name__ == '__main__':
    csv = "shuffled_posts.csv"
    data = load_data(csv)
    print(data['status_message'][1])
    processLanguage3(zip(data['status_message'][:100], data['status_published'][:100]))