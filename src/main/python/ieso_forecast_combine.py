import os
from XmlForecast import XmlForecast
from ieso_forecast_scraper import scrape

scrape('http://reports.ieso.ca/public/OntarioZonalDemand/',
       "../resources/data/demand/forecasts/previous_month/")

in_dir = "../resources/data/demand/forecasts/previous_month/"
out_file = "../resources/data/demand/forecasts/ieso_forecasts.csv"
with open(out_file, 'w') as f:
    print('Writing...')
    for line in XmlForecast(in_dir + os.listdir(in_dir)[0]).to_csv():
        f.write(line)

    for file in os.listdir(in_dir)[1:]:
        for line in XmlForecast(in_dir + file).to_csv()[1:]:
            f.write(line)
f.close()
print('Done')
