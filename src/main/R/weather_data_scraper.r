# Scrapes data from all stations that have an hourly observation after 2002-01-01 (start of demand and pricing data)
library(weathercan)
library(dplyr)

setwd('C:/Users/Patrick/OneDrive - The University of Western Ontario/Documents/Research/MITACS/Code/src/main/R/')

stations <- stations() %>% filter(end >= '2002-01-01', prov == 'ON', interval == 'hour')

for (station in 23:length(stations$station_id)){
  outfile <- paste('../resources/data/climate/ontario/csv_files/', as.character(stations$station_id[station]), '.csv', sep = '')
  print(outfile)
  data <- weather_dl(
    station_ids = stations$station_id[station],
    interval = 'hour',
    start = '2002-01-01'
  )
  write.csv(
    data,
    outfile
  )
}

## Last update 2022-02-09