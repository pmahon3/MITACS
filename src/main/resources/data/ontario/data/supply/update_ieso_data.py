from ..Scrapers.IesoSupply import *

demand = IesoSupply(url='http://reports.ieso.ca/public/GenOutputbyFuelHourly/')
demand.scrape_and_write(xml_out_dir='xml_files/',
                        csv_out_file='csv_files/ieso_supply.csv')
