# Poker face: predicting programming language

- by David Mitchell , Fred Lindsey, and Vasiliy Melkozerov Jul 25, 2020

<img src="https://www.science.org/do/10.1126/science.aaa6312/full/sn-texasholdemh.jpg" width="100%" height="300-">


## Project description:

What had happened was that we scarped github to see what words came up in different languages, now we gotta look for the big tings

The purpose and goal of this project is to be able to model programming languages based off of words used in project README.md

## Executive summary:



## Guiding questions:
- What are the most common words in READMEs?
- Are there bi-grams useful in finding programming languages?
- Do different programming languages use a different number of unique words?
- Are there any words that uniquely identify a programming language?

## Project Goals:
- Find leading words indicating language
- Build a predictive classification algorithm 

## Data Dictionary

|Feature|Datatype|Definition|
|:-------|:--------|:----------|
| repo | 418 non-null: object | Name of repository |
| language | 418 non-null: object | Programming language most used |
| readme_contents | 418 non-null: object | Repository repos |
| stemmed_content | 418 non-null: object | Stemmed words using PorterStemmer|
| lemmed_content | 418 non-null: object | Lemmed words using WordNetLemmatizer |


## Findings/ Takeaways

## How to reproduce:
1. Get your own env setup github api keys and 

2. Clone the repository

3. In your terminal where this file is stored run line python acquire and wait for the list to be done

4. Now that you're dataset is in a json format run the notebook
