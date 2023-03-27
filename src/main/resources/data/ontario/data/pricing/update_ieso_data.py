from bs4 import BeautifulSoup

import os
import requests
import re

url = 'http://reports.ieso.ca/public/PriceHOEPPredispOR/'
csv_dir = './csv_files/'
combined_outfile = 'ieso_price.csv'
combined_out = ''

reqs = requests.get(url)
soup = BeautifulSoup(reqs.text, 'html.parser')

print('Scraping price data...')
urls = []
for link in soup.find_all('a'):
    link_str = link.get('href')
    if 'csv' in link_str[-4:]:
        if re.match(r'.*([1-3][0-9]{3})', link_str[-8:-4]) is None:
            continue
        else:
            print(link_str)
            response = requests.get(url + link_str)
            with open(csv_dir + link_str, 'wb') as file:
                file.write(response.content)
            file.close()
            with open(csv_dir + link_str, 'a') as file:
                file.write('\n')
            file.close()

print('Writing price data...')
i = 0
for file in os.listdir('./csv_files/'):
    with open(csv_dir + file, 'r') as f:
        head = [next(f) for line in range(3)]
        if i != 0:
            f.readline()
        for line in f.readlines():
            combined_out = combined_out + line
        f.close()
    i = i+1

with open(combined_outfile, 'w') as f:
    f.write(combined_out)
    f.close()

reqs.close()
soup.clear()
print('Done')
