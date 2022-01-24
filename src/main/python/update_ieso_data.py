from IesoForecast import *
from IesoDemand import *

forecast = IesoForecast(url = 'http://reports.ieso.ca/public/OntarioZonalDemand/')
forecast.scrape_and_write(xml_out_dir = '../resources/data/demand/forecast/individual_files/',
                          csv_out_file = '../resources/data/demand/forecast/ieso_forecasts.csv')

demand = IesoDemand(url = 'http://reports.ieso.ca/public/Demand/')
demand.scrape_and_write(csv_out_dir = '../resources/data/demand/actual/individual_files/',
                        csv_out_file='../resources/data/demand/actual/ieso_demand.csv')
