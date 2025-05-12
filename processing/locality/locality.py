import numpy as np
import pandas as pd
import warnings

from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from edynamics.modelling_tools import Lag, Embedding
from edynamics.modelling_tools.projectors import WeightedLeastSquares
from edynamics.modelling_tools.kernels import Exponential

warnings.filterwarnings("ignore")

# ── 1) Load & prepare your full, hourly‐indexed DataFrame ───────────
df = (
    pd.read_csv(
        "/Users/pmahon/Research/Dynamics/MITACS/processing/clustering/csv/stationary_candidate.csv",
        index_col=0,
        parse_dates=True
    )
    .asfreq("h")  # ensures df.index.freq == 'H'
)

thetas = np.arange(0, 11)


# ── 2) Worker initializer: runs once per process ────────────────────
def init_worker(df_full, thetas_arg, dimension_arg, full_library_times):
    global embedding_global, thetas_global
    thetas_global = thetas_arg
    # build embedding against the *full* library
    lags = [Lag(variable_name="zscore", tau=-i) for i in range(dimension_arg)]
    embedding_global = Embedding(
        data=df_full,
        observers=lags,
        library_times=full_library_times
    )


# ── 3) Per‐timestamp worker function ────────────────────────────────
def compute_projection(time):
    proj = WeightedLeastSquares(kernel=Exponential(theta=0.0))
    errors = {}
    obs0 = embedding_global.data.at[time, "zscore"]

    for θ in thetas_global:
        proj.kernel.theta = θ
        out = proj.project(
            embedding=embedding_global,
            points=embedding_global.block.loc[time:time],
            steps=1, step_size=1, leave_out=True
        )["zscore"]

        if out.empty:
            # no data returned → record NaN
            errors[θ] = np.nan
        else:
            pred0 = out.iat[0]
            errors[θ] = abs(pred0 - obs0) / abs(obs0)

    return time, errors


# ── 4) Main parallel loop ───────────────────────────────────────────
if __name__ == "__main__":
    for daytype in ["weekday", "saturday", "sunday"]:
        # load the dimension of the embedding
        dims = pd.read_csv(f'../dimensions/params/results_{daytype}.csv', index_col=0)
        dim = dims.idxmax()[0]

        # a) pick your full library of times for embedding
        # use only data from after 2023-01-01
        full_library_times = df.index[df["daytype"] == daytype][10:-1]

        # b) randomly sample 80 percent of the full library times
        sampled = np.random.choice(full_library_times, size=int(0.8 * len(full_library_times)), replace=False)
        prediction_times = pd.DatetimeIndex(sampled)

        # c) prepare a results DataFrame
        results = pd.DataFrame(index=prediction_times, columns=thetas, dtype=float)

        # d) launch Pool with initializer using the full library
        with Pool(
                processes=cpu_count(),
                initializer=init_worker,
                initargs=(df, thetas, dim, full_library_times)
        ) as pool:
            it = pool.imap_unordered(compute_projection, prediction_times)
            for time, err_dict in tqdm(
                    it,
                    total=len(prediction_times),
                    desc=f"Processing {daytype}"
            ):
                for θ, err in err_dict.items():
                    results.at[time, θ] = err

        mean_err = results.values.mean()
        print(f"{daytype:>9} → mean error = {mean_err:.4f}")
        results.to_csv(f"./csv/results_{daytype}.csv")
