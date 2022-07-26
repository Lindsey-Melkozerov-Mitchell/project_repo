# Poker face: predicting programming language

- by David Mitchell , Fred Lindsey, and Vasiliy Melkozerov Jul 25, 2020

<img src="poker_nlp_banner.jpg" width="100%" height="300-">


## Project description:

What had happened was that we scarped github to see what words came up in different languages, now we gotta look for the big tings

The purpose and goal of this project is to be able to model programming languages based off of words used in project README.md

## Guiding questions:
- What are the most common words in READMEs?
- Which machine learning modeling technique would yeild the best results?
- Do different programming languages use a different number of unique words?
- Are there any words that uniquely identify a programming language?

## Project Goals:
- Find leading words indicating language
- Build a predictive classification algorithm 

## Takeaway:
- the best perfoming classifier was a Decision Tree Classifier with a max depth value of 9
- however, the most robust DTC model was max depth 5, with a train/val difference of 13%, vs the higher performing model's 18.7% train/val difference.
- Due to the higher reliability predicting performance on the out-of-sample validation data, I'm going to deploy the more robust DTC model with max depth 5 on the test data, for the final prediction
- The final DTC model performed with a prediction accuracy of 40% on test data, higher than initial performance on validate data. While this is a generally good indicator, more refinement is needed on this model before I can recommend it for deployment as a predictive.


## Data Dictionary

|Feature|Datatype|Definition|
|:-------|:--------|:----------|
| repo | 418 non-null: object | Name of repository |
| language | 418 non-null: object | Programming language most used |
| readme_contents | 418 non-null: object | Repository repos |
| stemmed_content | 418 non-null: object | Stemmed words using PorterStemmer|
| lemmed_content | 418 non-null: object | Lemmed words using WordNetLemmatizer |


## How to reproduce:
1. Get your own env setup github api keys and 

2. Clone the repository

3. In your terminal where this file is stored run line python acquire and wait for the list to be done

4. Now that you're dataset is in a json format run the notebook
