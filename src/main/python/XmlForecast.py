import os
import pandas as pd


class XmlForecast:

    def __init__(self, in_file):
        self.file = in_file
        self.xml_str = open(self.file).read()
        self.csv = None

    def to_csv(self):
        """ Convert xml data to a csv string
            Parameters:
            out_file -- where to save the csv

            Returns:
            csv string
            """
        out = "Date,Zone,Hour,Demand\n"

        for date in self.get_element_bodies("DeliveryDate", self.xml_str):
            zones = ['Ontario', 'East', 'West']
            date_data = self.get_text_between('<DeliveryDate>' + date, '</ZonalDemands>', self.xml_str)
            for zone in zones:
                zone_data = self.get_element_bodies(zone, date_data)[0]
                demands = self.get_element_bodies('Demand', zone_data)
                for demand in demands:
                    hour = self.get_element_bodies('DeliveryHour', demand)[0]
                    mw = self.get_element_bodies('EnergyMW', demand)[0]
                    out = out + date + "," + zone + "," + hour + "," + mw + "\n"

        return out

    def get_text_between(self, first, second, content):
        """ Get text between two strings of the xml_string
            Parameters:
            first -- first string
            second -- second string

            Returns:
            string between first and second string
            """
        return content.split(first)[1].split(second)[0]

    def get_element_bodies(self, element_name, content):
        """ Get text between all instances of an element tag
            Parameters:
            element_name -- the name of the element, e.g. 'Date' in <Date>...</Date>
            content -- string to extract elements from

            Returns:
            list of strings occurring between each instance of tag pair <element_name>...</element_name>
            """
        bodies = []
        for element in content.split("<" + element_name + ">")[1:]:
            bodies.append(element.split('</' + element_name + ">")[0])
        return bodies
