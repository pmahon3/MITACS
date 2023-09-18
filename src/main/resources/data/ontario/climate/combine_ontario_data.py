import os
import time
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
import numpy as np

# Parameters
out = ""
values = [
    "hmdx",
    "precip_amt",
    "pressure",
    "rel_hum",
    "temp",
    "temp_dew",
    "wind_chill",
    "wind_dir",
    "wind_spd",
]
start = datetime.strptime("2002-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
end = datetime.fromtimestamp(
    max(os.stat(root).st_mtime for root, _, _ in os.walk("./"))
).replace(microsecond=0, second=0, minute=0) - timedelta(hours=1)
span = end - start
span = int(span.total_seconds() / 3600)

# Combining and writing
print("Writing...")

# Data
combined_out = open("ontario/climate_data.csv", "w")
averaged_out = open("ontario/climate_data_averaged.csv", "w")

# Data containers
times = [
    (start + timedelta(hours=x)).strftime("%Y-%m-%d %H:%M:%S") for x in range(span)
]
measurements = np.zeros(
    shape=(span, len(values)),
    dtype=list(zip(values, ["f4" for i in range(len(values))])),
)
counts = np.zeros(
    shape=(span, len(values)),
    dtype=list(zip(values, ["f4" for i in range(len(values))])),
)

# Output data
i = 0
for file in os.listdir("ontario/csv_files"):
    print(file)

    station = np.genfromtxt(
        fname="ontario/csv_files/" + file, delimiter=",", names=True
    )

    with open("ontario/csv_files/" + file, "r") as f:
        if i != 0:
            f.readline()
        else:
            for line in f.readlines():
                out = out + line
        f.close()
out.close()
print("Done")
