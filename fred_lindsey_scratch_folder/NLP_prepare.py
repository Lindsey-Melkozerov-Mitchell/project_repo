# Imports

import unicodedata
import re
import json
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
import pandas as pd
import acquire

#____________________________________________________________________________________________________________________________________________________________

def prepare_blogs(Series):
    """ This function takes in a Series and applies a series of cleaning functions before stemming and lemmatizing the text.
    Args: Series
    Function: .lower, normalize, remove non-ASCII, stem, lem
    Returns: Cleaned, stemmed, lemmatized Series in a DF, along with original text"""
    original_content = []
    clean_content = []
    stemmed_content = []
    lemmed_content = []
    blogs_dict = {'content': original_content,
    'cleaned_content': clean_content,
    'stemmed_content': stemmed_content,
    'lemmed_content': lemmed_content}
    for i in range(0, len(Series)):
        content = Series[i]
        # add unaltered text to list 'original_content'
        original_content.append(content)
        # convert to lower case
        content = content.lower()
        # remove accented characters
        # unicode: removes character encoding incosistencies
        # .encode: converts resulting str chars to ASCII set. ignore errors will drop no ASCII chars
        # .decode turns the bytes object back into an str
        content = unicodedata.normalize('NFKD', content)\
        .encode('ascii', 'ignore')\
        .decode('utf-8', 'ignore')
        # remove non-standard alphanumeric characters
        content = re.sub(r"[^a-z0-9'\s]", '', content)
        # tokenization:
        # break words and punctuation left over into discrete units
        tokenizer = nltk.tokenize.ToktokTokenizer()
        tokenizer.tokenize(content, return_str=True)
        # add the tokenized text to the list 'clean_content'
        clean_content.append(content)
        # stems are the base of words, call: calls, called, calling
        # to stem, create the object first:
        ps = nltk.porter.PorterStemmer()
        # then apply to all words in the article
        stems = [ps.stem(word) for word in content.split()]
        content_stemmed = ' '.join(stems)
        #add stemmed output to list:
        stemmed_content.append(content_stemmed)
        # lemmatizing: reduces the word by removing the suffix (if applicable), but leaves a lexi cor word
        # how to lemmatize:
        wnl = nltk.stem.WordNetLemmatizer()
        lemmas = [wnl.lemmatize(word) for word in content.split()]
        content_lemmatized = ' '.join(lemmas)
        # add lemmed content to list:
        lemmed_content.append(content_lemmatized)
    df = pd.DataFrame(blogs_dict)
    return df

#____________________________________________________________________________________________________________________________________________________________



def prepare_shorts(Series):
    """ This function takes in a Series and applies a series of cleaning functions before stemming and lemmatizing the text.
    Args: Series
    Function: .lower, normalize, remove non-ASCII, stem, lem
    Returns: Cleaned, stemmed, lemmatized Series in a DF, along with original text"""
    original_content = []
    clean_content = []
    stemmed_content = []
    lemmed_content = []
    shorts_dict = {'content': original_content,
    'cleaned_content': clean_content,
    'stemmed_content': stemmed_content,
    'lemmed_content': lemmed_content}
    for i in range(0, len(Series)):
        content = Series[i]
        # add unaltered text to list 'original_content'
        original_content.append(content)
        # convert to lower case
        content = content.lower()
        # remove accented characters
        # unicode: removes character encoding incosistencies
        # .encode: converts resulting str chars to ASCII set. ignore errors will drop no ASCII chars
        # .decode turns the bytes object back into an str
        content = unicodedata.normalize('NFKD', content)\
        .encode('ascii', 'ignore')\
        .decode('utf-8', 'ignore')
        # remove non-standard alphanumeric characters
        content = re.sub(r"[^a-z0-9'\s]", '', content)
        # tokenization:
        # break words and punctuation left over into discrete units
        tokenizer = nltk.tokenize.ToktokTokenizer()
        tokenizer.tokenize(content, return_str=True)
        # add the tokenized text to the list 'clean_content'
        clean_content.append(content)
        # stems are the base of words, call: calls, called, calling
        # to stem, create the object first:
        ps = nltk.porter.PorterStemmer()
        # then apply to all words in the article
        stems = [ps.stem(word) for word in content.split()]
        content_stemmed = ' '.join(stems)
        #add stemmed output to list:
        stemmed_content.append(content_stemmed)
        # lemmatizing: reduces the word by removing the suffix (if applicable), but leaves a lexi cor word
        # how to lemmatize:
        wnl = nltk.stem.WordNetLemmatizer()
        lemmas = [wnl.lemmatize(word) for word in content.split()]
        content_lemmatized = ' '.join(lemmas)
        # add lemmed content to list:
        lemmed_content.append(content_lemmatized)
    df = pd.DataFrame(shorts_dict)
    return df