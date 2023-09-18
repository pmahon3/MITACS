import os
import pandas as pd


class XmlForecast:
    def __init__(self, in_file):
        self.file = in_file
        with open(self.file, "r") as f:
            self.xml_str = f.read()
        f.close()
        self.csv = None

    def to_csv(self):
        """Convert xml data to a csv string
        Parameters:
        out_file -- where to save the csv

        Returns:
        csv as a string
        """
        out = "Date,Zone,Hour,Demand,CreationDate\n"

        creation_date = self.get_element_bodies("CreatedAt")[0]

        for date in self.get_element_bodies("DeliveryDate"):
            zones = ["Ontario", "East", "West"]
            date_data = self.get_text_between(
                "<DeliveryDate>" + date, "</ZonalDemands>"
            )
            for zone in zones:
                zone_data = self.get_element_bodies(zone, date_data)[0]
                demands = self.get_element_bodies("Demand", zone_data)
                for demand in demands:
                    hour = self.get_element_bodies("DeliveryHour", demand)[0]
                    mw = self.get_element_bodies("EnergyMW", demand)[0]
                    out = (
                        out
                        + date
                        + ","
                        + zone
                        + ","
                        + hour
                        + ","
                        + mw
                        + ","
                        + creation_date
                        + "\n"
                    )

        return out

    def get_text_between(self, first, second, content=None):
        """Get text between two strings of the xml_string
        Parameters:
        first   --  first string
        second  --  second string
        content --  optional argument string in which to perform text extraction
                    default value is objects entire xml string

        Returns:
        string between first and second string
        """
        if content is None:
            content = self.xml_str
        return content.split(first)[1].split(second)[0]

    def get_element_bodies(self, element_name, content=None):
        """Get text between all instances of an element tag
        Parameters:
        element_name    --  the name of the element, e.g. 'Date' in <Date>...</Date>
        content         --  optional string to extract elements from
                            default value is objects entire xml string

        Returns:
        list of strings occurring between each instance of tag pair <element_name>...</element_name>
        """
        if content is None:
            content = self.xml_str
        bodies = []
        for element in content.split("<" + element_name + ">")[1:]:
            bodies.append(element.split("</" + element_name + ">")[0])
        return bodies
