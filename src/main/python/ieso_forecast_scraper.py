import requests
from bs4 import BeautifulSoup

def scrape(url, out_dir):
    print('Scraping...')
    reqs = requests.get(url)
    soup = BeautifulSoup(reqs.text, 'html.parser')

    urls = []
    for link in soup.find_all('a'):
        link_str = link.get('href')
        if 'v1' in link_str:
            print(link_str)
            response = requests.get(url + link_str)
            with open(out_dir + link_str, 'wb') as file:
                file.write(response.content)
