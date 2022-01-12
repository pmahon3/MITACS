
library('dplyr')
library('rEDM')

location = './main/resources/demand/aggregate/'
start = 2002
stop =2021
n = stop-start+1
years = seq(start,stop,1)

data = data.frame()
for (year in 1:n){
  file_str = paste0(location, 'PUB_Demand_', as.character(years[year]), '.csv')
  data = rbind(data, read.csv(file_str, skip=3, header = TRUE))
}

write.csv(data, file = paste0(location, 'PUB_Demand_combined.csv'))

location = './main/resources/demand/zonal/'
data = data.frame()
start = 2003
stop =2021
n = stop-start+1
years = seq(start,stop,1)
for (year in 1:n){
  file_str = paste0(location, 'PUB_DemandZonal_', as.character(years[year]), '.csv')
  data = rbind(data, read.csv(file_str, skip=3, header = TRUE))
}

write.csv(data, file = paste0(location, 'PUB_DemandZonal_combined.csv'))