library(curl)

stations <- read.csv("../climate/weather_station_zones.csv")

download <- function(start, end, freq, lat, long, param, format, out){
  str <- paste0("https://power.larc.nasa.gov/api/temporal/",
                freq,
                "/point?start=", start,
                "&end=", end,
                "&latitude=", lat,
                "&longitude=", long,
                "&community=ag&parameters=", param,
                "&format=", format,
                "&header=true&time-standard=lst")
  print(paste0(lat, ", ", long))
  curl_download(str, out)
  Sys.sleep(1)
}

stations %>% rowwise() %>% mutate(
  download(
    '20200101',
    '20201231',
    'daily',
    latitude,
    longitude,
    "ALLSKY_SFC_SW_DWN",
    "csv",
    paste0("./irradiance/", gsub(" ", "_", Name), "_2020", ".csv")
  )
)

