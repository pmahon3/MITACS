# Resources
## Overview
This directory contains all time series and geographic datasets. Time series datasets are preferably in comma separated value (csv) format and geographic data for power demand zones is in kml/kmx google earth files. 

## Power Demand Data

Contained in the [demand subdirectory](https://github.com/pmahon3/MITACS/tree/main/src/main/resources/demand) includes demand aggregated across Ontario and demand in sub-Ontario zones as defined by [IESO](https://www.ieso.ca/localContent/zonal.map/index.html) with hourly granularity. 

## Price Data

Contained in the [pricing subdirectory](https://github.com/pmahon3/MITACS/tree/main/src/main/resources/pricing), data are the the [Hourly Ontario Energy Price](https://www.ieso.ca/en/Power-Data/Data-Directory#Hourly-Ontario-Energy-Price-(HOEP)). Price is fixed across Ontario and does not vary by demand zones as described above. 

## Climate Data

Contained in the [climate subdirectory](https://github.com/pmahon3/MITACS/tree/main/src/main/resources/climate), data sampled or sourced from various weather stations accross Ontario. Solar irradiance data extracted from the [NASA Power Project](https://power.larc.nasa.gov/) and weather data from the [Government of Canada](https://climate.weather.gc.ca/historical_data/search_historic_data_e.html). Weather data is at daily granularity and irradiance data hourly. 
