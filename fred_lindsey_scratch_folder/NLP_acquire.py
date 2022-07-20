 # use the requests module
from requests import get
import os
from requests import get
from bs4 import BeautifulSoup
import pandas as pd

#'data-science/jobs-after-a-coding-bootcamp-part-1-data-science/',



def title_and_content_scrapes(use_cache=True):
    url = 'https://codeup.com/'

    headers = {'User-Agent': 'CodeUp Data Science Student2'} #make sure this is rock solid, otherwise you can get a 403 error

    endpoints = ['data-science/jobs-after-a-coding-bootcamp-part-1-data-science/','featured/what-jobs-can-you-get-after-a-coding-bootcamp-part-2-cloud-administration/',
    'tips-for-prospective-students/is-our-cloud-administration-program-right-for-you/', 'codeup-news/inclusion-at-codeup-during-pride-month-and-always/', 
    'tips-for-prospective-students/mental-health-first-aid-training/']

    filename = "title_and_content.csv"
    if os.path.isfile(filename) and use_cache:
        return pd.read_csv(filename)
    else:
        #set an empty list to store dicts
        dict_list = []
        for endpoint in endpoints:
            response = get(url+endpoint, headers=headers)
            #print(response)
            soup = BeautifulSoup(response.text, 'html.parser')
            title = soup.title.string.strip()
            content = soup.article.text.strip()
        # capture title and content values from the scraping
            title_and_content_dict = {'title': title,
                    'content': content}
            dict_list.append(title_and_content_dict)
            #print(endpoint)
        title_and_content_df = pd.DataFrame(dict_list)
    title_and_content_df.to_csv(filename)
    return title_and_content_df


#________________________________________________________________________________________________________________________________________________________


def get_news_articles(use_cache=True):
    filename = "shorts_scrape.csv"
    if os.path.isfile(filename) and use_cache:
        return pd.read_csv(filename)
    else:
        url = 'https://inshorts.com/en/read'
        headers = {'User-Agent': 'CodeUp Data Science Student2'}
        response = get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        # build the categories from inshorts.com/en/read. We can use list methods to custom format it too (lower, drop the misc category, etc)
        categories = [li.text.lower() for li in soup.select('li')][1:]
        #breakers: if all news is removed as the first item, and if india is changed to something else
        categories[0] = 'national'
        # set up an empty list, to house the dictionaries
        inshorts = []
        #loop through the categores
        for category in categories:
            url = 'https://inshorts.com/en/read/' + category
            headers = {'User-Agent': 'CodeUp Data Science Student2'}
            response = get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')

            titles = [span.text for span in soup.find_all('span', itemprop='headline')]
            contents = [div.text for div in soup.find_all('div', itemprop='articleBody')]
            print(titles)
            for i in range(len(titles)):
                article = {'title': titles[i],
                'content': contents[i],
                'category': category}
                print(titles)
                inshorts.append(article)
        inshorts_df = pd.DataFrame(inshorts)
        inshorts_df.to_csv(filename)
        return inshorts_df