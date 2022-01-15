import pandas as pd
import random as rnd
from pathlib import Path
from datetime import datetime
from pandas import DataFrame

inFile = "../resources/ontario_averaged_first_diff_2020.csv"
outFile = "../resources/ontario_averaged_first_diff_weekly_shuffle_2020.csv"
ts: DataFrame = pd.DataFrame(pd.read_csv(inFile, index_col=0))

ts['Weekday'] = ts['DATE'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').strftime('%A'))

for i in range(1,len(ts)):
    temp = ts.loc[i]
    idx = ts[ts['Weekday'] == temp['Weekday']].sample().index[0]
    ts.loc[i] = ts.loc[idx]
    ts.loc[idx] = ts.loc[i]

ts.to_csv(outFile)
