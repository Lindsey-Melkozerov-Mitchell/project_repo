# Poker finder (working title)

- by David Mitchell , Fred Lindsey, and Vasiliy Melkozerov Jul 25, 2020

<hr>

## Project description:
The purpose and goal of this project is to be able to model programming languages based off of words used in project README.md
---

## Guiding questions:
- What are the most common words in READMEs?
- Does the length of the README vary by programming language?
- Do different programming languages use a different number of unique words?
- Are there any words that uniquely identify a programming language?

## Project objectives:
- Find leading words indicating language
- A machine learning model that can guess the programming language


|Feature|Datatype|Definition|
|:_______|:________|:__________|
| repo | 500 non-null: object | Name of repository |
| language | 481 non-null: object | Programming language most used |
| readme_contents | 500 non-null: object | Repository repos |



## How to reproduce:
1. Clone the repository

Get your own env setup github api keys and 

In your terminal where this file is stored run line 

To adjust size of the search go into the acquire file and place the number of repo's you want to scrape, in this experiement we will be going with 500 repos to start

The data comes scraped from github based off of the function