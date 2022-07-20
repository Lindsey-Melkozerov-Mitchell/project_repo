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