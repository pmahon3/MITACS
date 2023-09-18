from ...Scrapers.IesoDemand import IesoDemand

demand = IesoDemand(url="http://reports.ieso.ca/public/Demand/")
demand.scrape_and_write(csv_out_dir="csv_files/", csv_out_file="ieso_demand.csv")
