from IesoForecast import IesoForecast

forecast = IesoForecast(url = 'http://reports.ieso.ca/public/OntarioZonalDemand/')
forecast.scrape_and_write(xml_out_dir = '../resources/data/demand/forecasts/previous_month/',
                          csv_out_file = '../resources/data/demand/forecasts/ieso_forecasts.csv')
