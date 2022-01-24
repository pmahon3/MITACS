import requests
from XmlForecast import *
import os
from bs4 import BeautifulSoup


class IesoForecast:
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

        urls = []
        for link in soup.find_all('a'):
            link_str = link.get('href')
            if '.xml' in link_str:
                if 'v' in link_str:
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
            f.write('Date,Zone,Hour,Demand,CreationDate\n')
            for file in os.listdir(in_dir):
                csv = iter(XmlForecast(in_dir + file).to_csv().splitlines(keepends=True))
                next(csv)
                for line in csv:
                    f.write(line)
            f.write('\n')
        f.close()
        print('Done')
