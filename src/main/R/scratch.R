library(dplyr)
library(rEDM)

in_file = '../resources/data/ieso_combined.csv'

ieso_data =  data.frame(read.csv(in_file))

ieso_data <- ieso_data %>%
  select('X', 'Time', 'Ontario.Demand', 'HOEP', 'Nuclear', 'Gas', 'Hydro', 'Wind', 'Solar', 'Biofuel') %>%
  mutate('HOEP' = as.double(purrr::pmap_chr(.l=list(pattern=',', replacement='', x=HOEP), .f=sub))) %>%
  mutate('time' = X) %>%
  select(-X)

fields = combn(x=c('Ontario.Demand', 'HOEP', 'Nuclear', 'Gas', 'Hydro', 'Wind', 'Solar', 'Biofuel'), m=2)
fields = rbind(fields, vector(mode='character', length=dim(fields)[2]))


for (i in 1:length(fields)){
  variables = fields[1:2,i]
  variables = c('time', variables)
  data <- ieso_data %>% select(variables)

  length = 10000
  lib_sizes = paste('1000', ' ', as.character(length %/% 1000 * 1000), ' ', '1000', sep='')

  fields[3,i] = CCM(
    dataFrame=data[(nrow(data)-length):nrow(data),],
    E=6,
    tau=-1,
    Tp=1,
    target=fields[1,i],
    columns=fields[2,i],
    libSizes=lib_sizes,
    sample=1,
    showPlot=True
  )
}
