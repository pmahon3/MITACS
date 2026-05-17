import os
import glob
import pandas as pd


def convert_csvs_to_pickles(
        input_folder="historical_csvs",
        output_folder="pickled_data"
):
    """
    Iterates over all CSVs in `input_folder`, loads each (skipping first 3 lines),
    and pickles the resulting DataFrame to `output_folder`.
    """

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Find all CSV files in input_folder
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))

    for csv_path in csv_files:
        # Derive a pickle name from the CSV file name
        csv_name = os.path.basename(csv_path)  # e.g. "PUB_Demand_2025.csv"
        base_name, _ = os.path.splitext(csv_name)  # "PUB_Demand_2025"
        pickle_name = f"{base_name}.pkl"
        pickle_path = os.path.join(output_folder, pickle_name)

        print(f"Reading {csv_name} -> {pickle_name}")

        # Skip the first 3 lines of metadata
        df = pd.read_csv(csv_path, skiprows=3)

        # Optionally transform Date + Hour into a single DateTime column (uncomment as needed)
        df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d")
        df['Hour'] = df['Hour'].astype(int)
        df['DateTime'] = df['Date'] + pd.to_timedelta(df['Hour'] - 1, unit='h')
        df.drop(columns=['Date', 'Hour'], inplace=True)

        # Pickle the DataFrame
        df.to_pickle(pickle_path)
        print(f"Pickled to {pickle_path}\n")


if __name__ == "__main__":
    convert_csvs_to_pickles()