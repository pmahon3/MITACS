import requests
import os
from bs4 import BeautifulSoup


class IesoDemand:
    def __init__(self, url):
        self.url = url
        self.csv_dir = None

    def scrape_and_write(self, csv_out_dir, csv_out_file):
        self.scrape(csv_out_dir)
        self.write_csv(csv_out_file)

    def scrape(self, csv_out_dir):
        self.csv_dir = csv_out_dir
        print('Scraping...')
        reqs = requests.get(self.url)
        soup = BeautifulSoup(reqs.text, 'html.parser')

        urls = []
        for link in soup.find_all('a'):
            link_str = link.get('href')
            if 'csv' in link_str[-4:]:
                if 'v' in link_str[:-4]:
                    continue
                else:
                    print(link_str)
                    response = requests.get(self.url + link_str)
                    with open(self.csv_dir + link_str, 'wb') as file:
                        file.write(response.content)
                    file.close()
        reqs.close()
        soup.clear()

    def write_csv(self, out_file, in_dir=None):
        if in_dir is None:
            in_dir = self.csv_dir
        with open(out_file, 'w') as f_out:
            print('Writing...')
            f_out.write('Date,Hour,Market Demand,Ontario Demand\n')
            for file in os.listdir(in_dir):
                with open(in_dir + file, 'r') as f_in:
                    lines = f_in.readlines()
                    for line in lines:
                        f_out.write(line)
                f_in.close()
            f_out.write('\n')
        f_out.close()
        print('Done')
