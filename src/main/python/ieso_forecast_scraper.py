import requests
from bs4 import BeautifulSoup

url = 'http://reports.ieso.ca/public/OntarioZonalDemand/'
outDir = "../resources/data/demand/forecasts/previous_month/"
reqs = requests.get(url)
soup = BeautifulSoup(reqs.text, 'html.parser')

urls = []
for link in soup.find_all('a'):
    link_str = link.get('href')
    if 'v1' in link_str:
        response = requests.get(url + link_str)
        with open(outDir + link_str, 'wb') as file:
            file.write(response.content)