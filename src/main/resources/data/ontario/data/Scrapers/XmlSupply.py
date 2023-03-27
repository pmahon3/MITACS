import os
import pandas as pd


class XmlSupply:

    def __init__(self, in_file):
        self.file = in_file
        with open(self.file, 'r') as f:
            self.xml_str = f.read()
        f.close()
        self.csv = None

    def to_csv(self):
        """ Convert xml data to a csv string
            Parameters:
            out_file -- where to save the csv

            Returns:
            csv as a string
            """
        out = "Date,Hour,Nuclear,Gas,Hydro,Wind,Solar,Biofuel,Total Output\n"

        creation_date = self.get_element_bodies("CreatedAt")[0]
        fields = set(self.get_element_bodies('Fuel'))
        for date in self.get_element_bodies("Day"):
            date_data = self.get_text_between('<Day>' + date, '</DailyData>')
            for hour in self.get_element_bodies('Hour', content=date_data):
                hour_data = self.get_text_between('<Hour>' + hour, '</HourlyData>', content=date_data)
                data = dict.fromkeys(fields, 0)
                for field in fields:
                    field_data = self.get_text_between(field, '</EnergyValue>', content=hour_data)
                    try:
                        data[field] = self.get_element_bodies('Output', field_data)[0]
                    except IndexError as error:
                        data[field] = ''
                        print(error)
                        print(date + ' ' + hour + ' ' + field)
                nuclear = data['NUCLEAR']
                gas = data['GAS']
                hydro = data['HYDRO']
                wind = data['WIND']
                solar = data['SOLAR']
                biofuel = data['BIOFUEL']

                values = date + ',' + hour + ',' + nuclear + ',' + gas + ',' + hydro + ',' + wind + ',' + solar + ',' \
                         + biofuel + '\n'

                out = out + values

        out = out + '\n'
        return out

    def get_text_between(self, first, second, content=None):
        """ Get text between two strings of the xml_string
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
        """ Get text between all instances of an element tag
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
            bodies.append(element.split('</' + element_name + ">")[0])
        return bodies
