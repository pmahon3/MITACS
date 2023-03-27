import re

import requests
import os
from bs4 import BeautifulSoup
from .XmlSupply import XmlSupply


class IesoSupply:
    def __init__(self, url):
        self.url = url
        self.xml_dir = None

    def scrape_and_write(self, xml_out_dir, csv_out_file):
        self.scrape(xml_out_dir)
        self.write_csv(csv_out_file)

    def scrape(self, xml_out_dir):
        self.xml_dir = xml_out_dir
        print('Scraping...')
        reqs = requests.get(self.url)
        soup = BeautifulSoup(reqs.text, 'html.parser')

        for link in soup.find_all('a'):
            link_str = link.get('href')
            year = link_str[-8:-4]
            if '.xml' in link_str:
                if re.match(r'.*([1-3][0-9]{3})', year) is None:
                    continue
                else:
                    print(link_str)
                    response = requests.get(self.url + link_str)
                    with open(self.xml_dir + link_str, 'wb') as file:
                        file.write(response.content)
                    file.close()
        reqs.close()
        soup.clear()

    def write_csv(self, out_file, in_dir=None):
        if in_dir is None:
            in_dir = self.xml_dir
        with open(out_file, 'w') as f:
            print('Writing...')
            f.write('Date,Hour,Nuclear,Gas,Hydro,Wind,Solar,Biofuel,Total Output\n')
            for file in os.listdir(in_dir):
                print(file)
                csv = iter(XmlSupply(in_dir + file).to_csv().splitlines(keepends=True))
                next(csv)
                for line in csv:
                    f.write(line)
            f.write('\n')
        f.close()
        print('Done')
