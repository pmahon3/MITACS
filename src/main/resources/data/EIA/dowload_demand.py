import requests
import csv

# API Endpoint
url = "https://api.eia.gov/v2/electricity/rto/region-sub-ba-data/data/?api_key=aDCqWFUTYVDdxqD8g1cHsYVqrxdlDPf9UusGS2Ba&frequency=hourly&data[0]=value&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"

# Make the API request
response = requests.get(url)
data = response.json()
# Print the keys within the 'response' key
print(data["response"].keys())

# To get a deeper understanding, print a snippet of the 'response' content
print(data["response"])
