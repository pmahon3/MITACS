library(dplyr)
library(rEDM)
library(lubridate)

setwd('C:/Users/Patrick/OneDrive - The University of Western Ontario/Documents/Research/MITACS/Code/src/main/R/')
forecast_in_file <- "../resources/data/demand/forecast/ieso_forecasts.csv"
actual_in_file <- "../resources/data/demand/actual/ieso_demand.csv"

# read in data
forecasts <- data.frame(read.csv(forecast_in_file))
actual <- data.frame(read.csv(actual_in_file))

actual <- actual %>% mutate('Date' = ymd_h(paste(Date, Hour, sep = " "))) %>%
  select(Date, Ontario.Demand) %>% rename(Demand = Ontario.Demand)

# get most recent predictions for Ontario demand
forecasts <- forecasts %>%
  mutate("CreationDate" = as.POSIXct(CreationDate, format = '%Y-%m-%dT%H:%M:%S')) %>%
  mutate('Date' = as.POSIXct(paste(Date,Hour, sep = " "), format = '%Y-%m-%d %H')) %>%
  filter(Zone == 'Ontario') %>%
  group_by(Date) %>%
  filter(CreationDate == max(CreationDate)) %>%
  arrange(Date) %>%
  ungroup() %>%
  rename(Forecast = Demand) %>%
  select(-CreationDate, -Hour, -Zone) %>%
  distinct()

data <- merge(actual, forecasts, by = 'Date', all = TRUE) %>%
  filter(is.na(Demand) == FALSE) %>%
  mutate('Forecast' = as.integer(Forecast))

write.csv(data, '../resources/data/processed_data_sets/temp/forecast_demand.csv')