import pandas as pd
import statsmodels.api as sm
from EdmSeries import EdmSeries

in_file = "../resources/data/processed_data_sets/ontario_averaged_2020.csv"

p = 6
d = 0
q = 6
tp = 1
tau = -7

ts = pd.DataFrame(pd.read_csv(in_file, index_col=[0]))
ts_arr = pd.DataFrame(ts['Demand']).to_numpy()
mod = sm.tsa.arima.ARIMA(ts_arr, order=(p, d, q))
res = mod.fit()

ts['ARMA(6)'] = res.predict()

series = EdmSeries(
    time_series=ts,
    minE=1,
    maxE=10,
    minTp=1,
    maxTp=10,
    minTau=1,
    maxTau=10
)
series.perform_embeddings(
    variables=['Time', 'ARMA(6)'],
    target='ARMA(6)',
    E=6,
    split=0.8,
    data_out_dir="../resources/output/previous_month/embed/ARMA_6/data/",
    drop_na=False,
    show_plot=False
)
series.embed_plotting("../resources/output/previous_month/embed/ARMA_6/plots/")
series.perform_smaps(
    variables=['Time', 'ARMA(6)'],
    target='ARMA(6)',
    tau=-1,
    split=0.8,
    data_out_dir="../resources/output/previous_month/nl/ARMA_6/data/",
    drop_na=False,
    show_plot=False
)
series.smaps_plotting(plot_out_dir="../resources/output/previous_month/nl/ARMA_6/plots/")