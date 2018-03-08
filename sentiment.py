# -*- coding: utf-8 -*-
import re
import os
import glob
import nltk
import itertools
import collections
import pandas as pd
from collections import Counter
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

#Set length of word combinations for use in counters.
filesentiment = {}
os.chdir(compdir)
print("Loading dictionary......")
#Get all files in the directory loaded into the corpus
for file in glob.glob("*.txt"):
    f = open(file)
    txtstream = f.read()
    txtstream2 = remove_punctuation(txtstream)
    filesentiment.update({file : sid.polarity_scores(txtstream2)})
    f.close()

print("Creating Dataframe......")
ods = collections.OrderedDict(filesentiment.items())
dfSentiment = pd.DataFrame.from_dict(ods, orient='index').reset_index()
dfSentiment = dfSentiment.rename(columns={'index':'Files Analyzed', 'pos':'Positive', 'neu':'Neutral' ,'neg':'Negative'})
dfSentiment = dfSentiment.drop('compound', 1)
dfSentiment = dfSentiment[['Files Analyzed', 'Positive', 'Neutral' ,'Negative']]
dfSentiment['Score'] = (dfSentiment['Positive'] + dfSentiment['Neutral']) - dfSentiment['Negative']
dfSentiment.sort_values(["Score"], inplace=True, ascending=False)
print("Writing output......")
os.chdir('..')
dfSentiment.to_html(open(sentimenthtml, 'w'))
