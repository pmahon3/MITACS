import pandas as pd
import os
import xml.etree.ElementTree as ET

in_dir = "../resources/data/demand/forecasts/previous_month/"

for file in os.listdir(in_dir):
    xml_data = pd.read_xml(in_dir + file)

