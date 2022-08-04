import requests
from requests import get
from bs4 import BeautifulSoup
from env import github_token, github_username
import acquire
import os

def get_poker_repositories(df):
    url_to_call = 'https://github.com/search?q=Poker&type=repositories'
    response = requests.get(url_to_call, headers = {"Authorization": f"token {github_token}", "User-Agent": github_username})
    response_code = response.status_code
    if response_code != 200:
        print('Error occurred')

    html_content = response.content
    soup = BeautifulSoup(html_content, 'html.parser')


    soup.find_all('a',class_='v-align-middle')

    soup.find_all('a',class_='v-align-middle')[0].text

    repos=[a.text for a in soup.find_all('a',class_='v-align-middle')]
    
    return df