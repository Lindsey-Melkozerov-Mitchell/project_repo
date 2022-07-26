"""
A module for obtaining repo readme and language data from the github API.
Before using this module, read through it, and follow the instructions marked
TODO.
After doing so, run it like this:
    python acquire.py
To create the `data.json` file that contains the data.
"""
import os
import json
from typing import Dict, List, Optional, Union, cast
import requests 
from requests import get
from bs4 import BeautifulSoup
import time

from env import github_token, github_username

# TODO: Make a github personal access token.
#     1. Go here and generate a personal access token: https://github.com/settings/tokens
#        You do _not_ need select any scopes, i.e. leave all the checkboxes unchecked
#     2. Save it in your env.py file under the variable `github_token`
# TODO: Add your github username to your env.py file under the variable `github_username`
# TODO: Add more repositories to the `REPOS` list below.

def get_repos(n):
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
        all_repos.append(repo)
        print(f'\rFetching page {page} of {n-1} {url}', end='')
    return all_repos

REPOS = get_repos(5000)

# REPOS = [
#     "dickreuter/Poker",     #https://github.com/dickreuter/Poker
#     "pokerregion/poker",         #https://github.com/pokerregion/poker
#     "ishikota/PyPokerEngine",              #https://github.com/ishikota/PyPokerEngine
#     "andrewprock/pokerstove",               #https://github.com/andrewprock/pokerstove
#     "mdp/JsPoker",          #https://github.com/mdp/JsPoker
#     "Jeremiah9000/Poker-with-Python",           #https://github.com/Jeremiah9000/Poker-with-Python
#     "ginuerzh/poker",           #https://github.com/ginuerzh/poker
#     "rundef/node-poker-odds-calculator",        #https://github.com/rundef/node-poker-odds-calculator
#     "datamllab/rlcard",             #https://github.com/datamllab/rlcard
#     "Wizehive/firepoker",           #https://github.com/Wizehive/firepoker
#     "HHSmithy/PokerHandHistoryParser"          #https://github.com/HHSmithy/PokerHandHistoryParser
# ]

headers = {"Authorization": f"token {github_token}", "User-Agent": github_username}

if headers["Authorization"] == "token " or headers["User-Agent"] == "":
    raise Exception(
        "You need to follow the instructions marked TODO in this script before trying to use it"
    )


def github_api_request(url: str) -> Union[List, Dict]:
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
    return [process_repo(repo) for repo in REPOS]


if __name__ == "__main__":
    data = scrape_github_data()
    json.dump(data, open("data.json", "w"), indent=1)