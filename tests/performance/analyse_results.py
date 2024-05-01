# Standard imports
from pathlib import Path

# Third party imports
import pandas as pd

# Project imports
from utils import plot_results, reformat_average_dataframe, sigma_clip

# Parameters for tests
clip = True
ignore_nan = True
record = True

# Set up the paths
performance_path = Path("tests/performance")
data_path = performance_path / "data"
record_path = data_path / "average.csv"

# Iterate through directories in data_path
averages = pd.DataFrame()
for item in data_path.iterdir():
    if item.is_dir():
        path_to_csv = item / "data.csv"
        if path_to_csv.is_file():
            print(f"\n----Making a new row from {item.name}----\n")
            df = pd.read_csv(path_to_csv, index_col=False)
            if clip:
                df = sigma_clip(df)
            new_row = {column: df[column].mean() for column in df.columns}
            new_row["time_stamp"] = item.name
            new_row = pd.DataFrame(new_row, index=[0])
            new_row.drop(
                new_row.columns[new_row.columns.str.contains("unnamed", case=False)],
                axis=1,
                inplace=True,
            )
            averages = pd.concat([averages, new_row], ignore_index=True)
        else:
            print(f"No data.csv in {item.name}")

# Print to screen for debugging
print(averages.columns)
print(averages.head)

# Reformat the dataframe
averages.to_csv(record_path)
list_dataframes = reformat_average_dataframe(averages)

# Save results
for dataframe in list_dataframes:
    print(f"\n----{dataframe.name}----\n")
    print(dataframe)
    plot_results(dataframe, data_path, date=True)
    plot_results(dataframe, data_path, date=True, log=True)
