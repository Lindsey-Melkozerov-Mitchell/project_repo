import os
import json
from typing import Dict, List, Optional, Union, cast
from requests import get
from bs4 import BeautifulSoup
import time
import acquire
from acquire import scrape_github_data
import pandas as pd
from env import github_token, github_username
import unicodedata
import re
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
# ignore warnings
import warnings
warnings.filterwarnings("ignore")
# get the basics for math and visuals
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# add the tools for classification reports
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
# pull in Decision Tree classifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
# pull in Random Forest classifer
from sklearn.ensemble import RandomForestClassifier
# pull in KNN classifer
from sklearn.neighbors import KNeighborsClassifier
# pull in Logistic Regression classifer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
# TF-IDF via scikit learn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
#________________________________________________________________________________________________________________________________

# ACQUIRE MODULE

#________________________________________________________________________________________________________________________________

"""
A module for obtaining repo readme and language data from the github API.
Before using this module, read through it, and follow the instructions marked
TODO.
After doing so, run it like this:
    python acquire.py
To create the `data.json` file that contains the data.
"""
import os
import requests
import json
from typing import Dict, List, Optional, Union, cast
from requests import get
from bs4 import BeautifulSoup
import time
import pandas as pd

from env import github_token, github_username
#note

# TODO: Make a github personal access token.
#     1. Go here and generate a personal access token: https://github.com/settings/tokens
#        You do _not_ need select any scopes, i.e. leave all the checkboxes unchecked
#     2. Save it in your env.py file under the variable `github_token`
# TODO: Add your github username to your env.py file under the variable `github_username`
# TODO: Add more repositories to the `REPOS` list below.

def get_repos(n, use_cache=True):
    filename = "poker_scrape.csv"
    if os.path.isfile(filename) and use_cache:
        # .values returns a list of values from Series, instead of
        # a Series, which this acquire cannot process.
        return pd.read_csv(filename).values
    all_repos = []
    for page in range(1, n):
        url = f'https://github.com/search?p={page}&q=poker&type=Repositories'
        headers = {"Authorization": f"token {github_token}", "User-Agent": github_username}
        while True:
            response = get(url, headers=headers)
            if response.ok:
                break
            else:
                time.sleep(15)
        soup = BeautifulSoup(response.text, 'html.parser')
        repo = [a.text for a in soup.find_all('a', class_='v-align-middle')]
        all_repos = all_repos + repo
        print(f'\rFetching page {page} of {n-1} {url}', end='')
    all_repos_series = pd.Series(all_repos)
    all_repos_series.to_csv(filename, index=False)
    return all_repos




#if headers["Authorization"] == "token " or headers["User-Agent"] == "":
#    raise Exception(
#        "You need to follow the instructions marked TODO in this script before trying to use it"
#    )


def github_api_request(url: str) -> Union[List, Dict]:
    headers = {"Authorization": f"token {github_token}", "User-Agent": github_username}
    response = requests.get(url, headers=headers)
    response_data = response.json()
    if response.status_code != 200:
        raise Exception(
            f"Error response from github api! status code: {response.status_code}, "
            f"response: {json.dumps(response_data)}"
        )
    return response_data


def get_repo_language(repo: str) -> str:
    url = f"https://api.github.com/repos/{repo}"
    repo_info = github_api_request(url)
    if type(repo_info) is dict:
        repo_info = cast(Dict, repo_info)
        if "language" not in repo_info:
            raise Exception(
                "'language' key not round in response\n{}".format(json.dumps(repo_info))
            )
        return repo_info["language"]
    raise Exception(
        f"Expecting a dictionary response from {url}, instead got {json.dumps(repo_info)}"
    )


def get_repo_contents(repo: str) -> List[Dict[str, str]]:
    url = f"https://api.github.com/repos/{repo}/contents/"
    contents = github_api_request(url)
    if type(contents) is list:
        contents = cast(List, contents)
        return contents
    raise Exception(
        f"Expecting a list response from {url}, instead got {json.dumps(contents)}"
    )


def get_readme_download_url(files: List[Dict[str, str]]) -> str:
    """
    Takes in a response from the github api that lists the files in a repo and
    returns the url that can be used to download the repo's README file.
    """
    for file in files:
        if file["name"].lower().startswith("readme"):
            return file["download_url"]
    return ""


def process_repo(repo: str) -> Dict[str, str]:
    """
    Takes a repo name like "gocodeup/codeup-setup-script" and returns a
    dictionary with the language of the repo and the readme contents.
    """
    contents = get_repo_contents(repo)
    readme_download_url = get_readme_download_url(contents)
    if readme_download_url == "":
        readme_contents = ""
    else:
        readme_contents = requests.get(readme_download_url).text
    return {
        "repo": repo,
        "language": get_repo_language(repo),
        "readme_contents": readme_contents,
    }


def scrape_github_data() -> List[Dict[str, str]]:
    """
    Loop through all of the repos and process them. Returns the processed data.
    """
    REPOS = get_repos(51)
    REPOS = [re for rep in REPOS for re in rep]
    print(REPOS)
    return [process_repo(repo) for repo in REPOS]


if __name__ == "__main__":
    REPOS = get_repos(51)
    data = scrape_github_data()
    json.dump(data, open("data.json", "w"), indent=1)
    
def main():
    data = scrape_github_data()
    json.dump(data, open("data.json", "w"), indent=1)
#________________________________________________________________________________________________________________________________

# PREPARE MODULE

#________________________________________________________________________________________________________________________________


def get_NLP_df():
    df = pd.read_json("data.json")
    return df

#________________________________________________________________________________________________________________________________
def remove_stopwords(string, extra_words = [], exclude_words = []):
    additional_stopwords = ['github', 'http', 'code']
    #nltk.download('wordnet')
    #nltk.download('stopwords')
    stopword_list = stopwords.words('english') + additional_stopwords
    stopword_list = set(stopword_list) - set(exclude_words)
    stopword_list = stopword_list.union(set(extra_words))
    words = string.split()
    filtered_words = [word for word in words if word not in stopword_list]
    string_without_stopwords = ' '.join(filtered_words)
    return string_without_stopwords

#________________________________________________________________________________________________________________________________

# prepare the DF (stem, lemmatize) for processing 

def prepare_poker(Series):
    """ This function takes in a Series and applies a series of cleaning functions before stemming and lemmatizing the text.
    Args: Series
    Function: .lower, normalize, remove non-ASCII, stem, lem
    Returns: Cleaned, stemmed, lemmatized Series in a DF, along with original text"""
    original_content = []
    #clean_content = []
    stemmed_content = []
    lemmed_content = []
    repos_dict = {'content': original_content,
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
        # remove stopwords
        stopwords = nltk.corpus.stopwords.words('english')
        content = remove_stopwords(content)
        # tokenization:
        # break words and punctuation left over into discrete units
        tokenizer = nltk.tokenize.ToktokTokenizer()
        tokenizer.tokenize(content, return_str=True)
        # add the tokenized text to the list 'clean_content'
        #clean_content.append(content)
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
    df = pd.DataFrame(repos_dict)
    return df


# combine the processed dataframe with the original, to include lables for the READMEs

def get_labeled_df():
    df = get_NLP_df()
    df = df.drop_duplicates(subset='readme_contents', keep='first', inplace=False, ignore_index=False)
    # drop nulls from the df
    df = df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
    # reset the index after dropping nulls, to allow sequential parsing to run without errors
    df = df.reset_index()
    # run the df through the prepare_function
    processed_df = prepare_poker(df.readme_contents)
    df = df.merge(processed_df, how='left', left_on='readme_contents', right_on='content')
    return df

#________________________________________________________________________________________________________________________________

# MODEL PREP

#_____________________________________________________________________________

def train_validate_test_split(df, target, seed=123):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed, 
                                            stratify=df[target])
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed,
                                       stratify=train_validate[target])
    return train, validate, test

#_____________________________________________________________________________

def remove_under_represented_values():
    df = get_labeled_df()
    values =['ShaderLab', 'Pascal', 'Apex', 'Elm', 'Julia', 'F#', 'Kotlin', 'Shell', 
    'CoffeeScript', 'Lua', 'Svelte', 'Roff', 'Vue', 'Dart']
    df = df[df.language.isin(values) == False]
    return df

#_____________________________________________________________________________

def split_and_vectorize(df):
    # create the object
    tfidf = TfidfVectorizer()
    # rename 'language' column to avoid duplication with encoded variables
    df = df.rename(columns={'language': 'programming_language_99'})
    # fit the vectorizer on all the lemmed_content
    X = tfidf.fit_transform(df.lemmed_content)
    y = df.programming_language_99
    # make the df out of the sparse matrix from TDIDF vectorizer
    tfidf_df = pd.DataFrame(X.todense(), columns = tfidf.get_feature_names())
    # concats with language lables
    encoded_df = pd.concat([tfidf_df, df.programming_language_99.reset_index()], axis=1)
    #split the data into train, validate, test segments
    train, validate, test = train_validate_test_split(encoded_df, 'programming_language_99')
    # print the split sizes as an internal check
    print(train.shape[0], validate.shape[0], test.shape[0])
    return train, validate, test
#________________________________________________________________________________________________________________________________

# MODELING

#_____________________________________________________________________________


def build_X_and_y(train, validate, test):
    X_train = train.drop(columns=['index', 'programming_language_99'])
    y_train = train.programming_language_99

    X_validate = validate.drop(columns=['index','programming_language_99'])
    y_validate = validate.programming_language_99

    X_test = test.drop(columns=['index', 'programming_language_99'])
    y_test = test.programming_language_99
    return X_train, y_train, X_validate, y_validate, X_test, y_test
#_____________________________________________________________________________

# Decision Tree Classifier

def DTC_model_and_df(X_train, y_train, X_validate, y_validate):
    max_depth_list = []
    reports = []

    # write the for loop to sequentially loop through the values for i,
    # setting i as the value, or inverse value, for 
    for i in range(1, 11):
            # create the decision tree object with desired hyper-parameters:
            clf = DecisionTreeClassifier(max_depth=i)
        
            # fit the decision tree to the training data:
            clf = clf.fit(X_train, y_train)
        
            #make predictions:
            language_prediction = pd.DataFrame(clf.predict(X_train))
            
            # Predict probability
            language_prediction_proba = pd.DataFrame(clf.predict_proba(X_train))
            
            # compute the estimate accuracy
            train_set_accuracy = clf.score(X_train, y_train)
        
            #evaluate on out-of-sample-date
            validate_set_accuracy = clf.score(X_validate, y_validate)
        
            max_depth_list.append({
                                'max_depth': i,
                                'training_accuracy': train_set_accuracy,
                                'validate_accuracy': validate_set_accuracy
                                        })
    df = pd.DataFrame(max_depth_list)
    df['difference'] = (df.training_accuracy - df.validate_accuracy)
    return df.sort_values(['validate_accuracy','difference'], ascending=[False, True]).head(5)

def visualize_DTC(df):
    plt.figure(figsize=(12, 9))
    sns.set(font_scale = 1.3)
    df[['training_accuracy', 'validate_accuracy', 'difference' ]].plot()
    plt.ylabel("accuracy")
    plt.xlabel("model number")
    # plt.vlines(x=[10], ymin=0, ymax=1, colors='r', linestyles='dashed')
    plt.title("DTC Models, performance on Train and Validate")
    plt.show()

#_____________________________________________________________________________

# Random Forest Classifier

def RFC_model_and_df(X_train, y_train, X_validate, y_validate):
    # create an empty list to store the dictionaries
    max_depth_and_leaf_samples_list = []
    # write the for loop to sequentially loop through the values for i,
    # setting i as the value, or inverse value, for 
    for i in range(1, 16):
        for j in range(1, 16):
            # create the random forest object with desired hyper-parameters:
            rf = RandomForestClassifier(max_depth=i, min_samples_leaf=j)
            # fit the random forest to the training data:
            rf.fit(X_train, y_train)
            # Evaluate importance, or weight, of each feature.
            rf.feature_importances_
            # Classify each passenger by its estimated survival.
            y_pred = rf.predict(X_train)
            # Estimate the probability of survival, using the training data.
            y_pred_proba = rf.predict_proba(X_train)
            # compute the estimate accuracy
            train_set_accuracy = rf.score(X_train, y_train)
            #evaluate on out-of-sample-date
            validate_set_accuracy = rf.score(X_validate, y_validate)
            max_depth_and_leaf_samples_list.append({
                                'max_depth': i,
                                'min_samples_leaf': j,
                                'training_accuracy': train_set_accuracy,
                                'validate_accuracy': validate_set_accuracy,
                                'difference': (train_set_accuracy - validate_set_accuracy)
                                        })
    df = pd.DataFrame(max_depth_and_leaf_samples_list)
    return df.sort_values(['validate_accuracy', 'difference'], ascending = [False, True]).head(3)

def visualze_RFC(df):    
    plt.figure(figsize=(15, 10))
    sns.set(font_scale = 1.3)
    df[['training_accuracy', 'validate_accuracy', 'difference' ]].plot()
    plt.title("Random Forest Model Accuracies")
    plt.ylabel("accuracy")
    plt.xlabel("model number")
    # plt.vlines(x=[163], ymin=0, ymax=1, colors='r', linestyles='dashed')
    plt.title("RFC Models, performance on Train and Validate")
    plt.show()



#_____________________________________________________________________________

# Logistic Regression Classifier

def LRC_model_and_df(X_train, y_train, X_validate, y_validate):
    log_regress_outputs = []
    i = 1
    # from sklearn.linear_model import LogisticRegression
    logit = LogisticRegression(C=i, random_state=123, intercept_scaling=1, solver='lbfgs')
    # Fit the Logistic Regression model
    logit.fit(X_train, y_train)
    # Get the predictions from the Logistic Regression Model
    y_pred = logit.predict(X_train)
    y_pred_proba = logit.predict_proba(X_train)
    # compute the estimate accuracy
    train_set_accuracy = logit.score(X_train, y_train)
    #evaluate on out-of-sample-data
    validate_set_accuracy = logit.score(X_validate, y_validate)
    log_regress_outputs.append({
                                'c_values': i,
                                'training_accuracy': train_set_accuracy,
                                'validate_accuracy': validate_set_accuracy,
                                'train_val_diff': (train_set_accuracy - validate_set_accuracy)
                                    })
    df = pd.DataFrame(log_regress_outputs)
    return df.sort_values(['validate_accuracy', 'train_val_diff'], ascending = [False, True]).head(3)

def visualze_LRC(df):    
    plt.figure(figsize=(15, 10))
    sns.set(font_scale = 1.3)
    df[['training_accuracy', 'validate_accuracy', 'train_val_diff' ]].plot()
    plt.title("Logistic Regression Model Accuracies")
    plt.ylabel("accuracy")
    plt.xlabel("model number")
    # plt.vlines(x=[163], ymin=0, ymax=1, colors='r', linestyles='dashed')
    plt.title("RFC Models, performance on Train and Validate")
    plt.show()
