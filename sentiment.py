# -*- coding: utf-8 -*-
import re
import os
import glob
import nltk
import itertools
import collections
import pandas as pd
from math import log
from nltk import ngrams
from collections import Counter
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


sentimenthtml = "sentiment.html"
compdir  = 'forprocessing/'

#Initialize Sentiment analyzer
sid = SentimentIntensityAnalyzer()
def remove_punctuation(text):
    # Removes all punctuation and conotation from the string and returns a 'plain' string
    punctuation2 = '-&'+'®©™€â´‚³©¥ã¼•ž®è±äüöž!@#Â“§$%^*()î_+€$=¿{”}[]:«;"»\â¢|<>,.?/~`0123456789\n'
    for sign in punctuation2:
        text = text.replace(sign, " ").lower()
    return text

def entities(text):
    # Breaks down stream into identified entities
    return nltk.pos_tag(word_tokenize(text))

#Set length of word combinations for use in counters.
dEntities = {}
filesentiment = {}
corpus = []
entities_list = []
file_list = []
os.chdir(compdir)
print("Loading corpus......")
#Get all files in the directory loaded into the corpus
for file in glob.glob("*.txt"):
    file_list.append(file)
    f = open(file)
    txtstream = f.read()
    txtstream2 = remove_punctuation(txtstream)
    corpus.append(txtstream)
    filesentiment.update({file : sid.polarity_scores(txtstream2)})
    dEntities.update({file : entities(txtstream)})
    entities_list.append(entities(txtstream))
    f.close()


ods = collections.OrderedDict(filesentiment.items())
dfSentiment = pd.DataFrame.from_dict(ods, orient='index').reset_index()
dfSentiment = dfSentiment.rename(columns={'index':'Files Analyzed', 0:'neg', 0:'neu', 0:'pos', 0:'compound'})
dfSentiment.sort_values(["compound"], inplace=True, ascending=False)
dfSentiment.at['Averages', 'neg'] = dfSentiment['neg'].mean()
dfSentiment.at['Averages', 'neu'] = dfSentiment['neu'].mean()
dfSentiment.at['Averages', 'pos'] = dfSentiment['pos'].mean()
SentScore = dfSentiment['compound'].mean()
dfSentiment.at['Averages', 'compound'] = SentScore


os.chdir('..')
dfSentiment.to_html(open(sentimenthtml, 'w'))