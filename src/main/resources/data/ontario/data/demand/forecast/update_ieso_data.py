from ...Scrapers.IesoForecast import IesoForecast

forecast = IesoForecast(url = 'http://reports.ieso.ca/public/OntarioZonalDemand/')
forecast.scrape_and_write(xml_out_dir ='xml_files/',
                          csv_out_file ='csv_files/ieso_forecasts.csv')
