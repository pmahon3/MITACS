import pandas as pd
import matplotlib
import multiprocessing
import matplotlib.pyplot as plt
import ray
import numpy as np

from edynamics.modelling_tools import Lag, Embedding
from edynamics.modelling_tools.estimators import dimensionality

from ray.util.multiprocessing import Pool

matplotlib.use("TkAgg")

ray.init(log_to_driver=False)

pool = Pool(multiprocessing.cpu_count())

# ── Load your data ────────────────────────────────────────────────
# Load the CSV file into a DataFrame
df = pd.read_csv("/Users/pmahon/Research/Dynamics/MITACS/processing/clustering/csv/stationary_candidate.csv", index_col=0)
df.index = pd.DatetimeIndex(df.index)
df = df.asfreq("h")  # Set frequency to hourly

# ── 0.  Setup  ──────────────────────────────────────────────────────────

# Build the lagged embedding for each day type

for daytype in ["weekday", "saturday", "sunday"]:
    # Get the library_indices for the current day type
    library_indices = df.index[df["daytype"] == daytype][10:]
    # Randomly select 80 percent of the library_indices for the prediction but avoid 7 PM
    sampled = np.random.choice(library_indices[:-1], size=int(0.2 * len(library_indices)), replace=False)
    prediction_times = pd.DatetimeIndex(sampled)
    # Remove 7 PM from the prediction times
    prediction_times = prediction_times[prediction_times.hour != 19]

    embedding = Embedding(
        data=df,
        observers=[Lag(variable_name="zscore", tau=0)],
        library_times=library_indices
    )

    result = dimensionality(
        embedding=embedding,
        target="zscore",
        steps=1,
        step_size=1,
        times=prediction_times,
        max_dimensions=10,
        compute_pool=pool,
        verbose=False
    )

    # Save the results to csv file
    result.to_csv(f"./params/results_{daytype}.csv", index=True)

    plt.plot(result)
    plt.title(f"Dimensionality for {daytype.capitalize()}")
    # X axis is number of nearest neighbors used in the K Nearest Neighbors projection
    plt.xlabel("K Nearest Neighbors Used")
    # Y axis is the pearson correlation coefficient between the predicted and observed values
    plt.ylabel("Pearson Correlation")
    plt.savefig(f"./plots/dimensionality_{daytype}.png")
    plt.cla()
