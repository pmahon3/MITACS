library(dplyr)

# data file
setwd('C:/Users/Patrick/OneDrive - The University of Western Ontario/Documents/Research/MITACS/Code/src/main/R/')
forecast_in_file <- "../resources/data/demand/forecast/csv_files/ieso_forecasts.csv"
actual_in_file <- "../resources/data/demand/actual/ieso_demand.csv"
bins <- 100

# read in data
forecasts <- data.frame(read.csv(forecast_in_file))
actual <- data.frame(read.csv(actual_in_file))
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

actual <- actual %>%
  select(c('Date', 'Hour', 'Ontario.Demand')) %>%
  mutate('Date' = as.POSIXct(paste(Date, Hour, sep = " "), format = '%Y-%m-%d %H')) %>%
  select(-Hour) %>%
  rename(Demand = 'Ontario.Demand') %>%
  distinct()

data <- merge(forecasts, actual, by = 'Date')
data <- data %>%
  mutate('Forecast' = as.integer(Forecast)) %>%
  mutate('Demand' = as.integer(Demand)) %>%
  mutate('Residual' = lead(Demand - Forecast, n=1)) %>%
  mutate('Rank' = ntile(Demand,bins))

stdev_r <- sd(data$Residual, na.rm = TRUE)

binned <- data %>%
  group_by(Rank) %>%
  summarize_at(c('Demand', 'Residual'), mean, na.rm = TRUE) %>%
  mutate('Residual^2' = Residual*Residual)

idx <- mean(binned$`Residual^2`, na.rm = TRUE) / (sd(binned$Residual, na.rm = TRUE )*stdev_r)

plot(binned$Demand, binned$Residual, xlab = 'x_t', ylab = 'r_(t+1)')
print(idx)