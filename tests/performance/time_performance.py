# Standard imports
import argparse
from pathlib import Path

# Third party imports
import pandas as pd

# Project imports
import twinlab as tl

# Importing timing functions
from utils import create_timestamp_directory, record_results, time_function, time_method

# Read command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--record", required=False, help="bool to record results", default=True
)
parser.add_argument(
    "--num_iterations", required=False, help="number of iterations to run", default=10
)
args = parser.parse_args()
record = args.record
num_iterations = int(args.num_iterations)

# General
verbose = True

### Parameters ###

# Path
resources_path = Path(__file__).resolve().parents[4] / "resources"

# Training
dataset_id = "biscuits_dataset"
emulator_id = "biscuits_emulator"
inputs = ["Pack price [GBP]", "Number of biscuits per pack"]
outputs = ["Number of packs sold", "Profit [GBP]"]
train_test_ratio = 0.8
seed = 123
dataset_path = resources_path / "datasets" / "biscuits.csv"
df = pd.read_csv(dataset_path)

# Prediction/sample
num_samples = 5
eval_path = resources_path / "campaigns" / "biscuits" / "eval.csv"
eval_df = pd.read_csv(eval_path)

# Recommend
num_points = 5
acq_func = "qExpectedImprovement"

# Calibration
obs_path = resources_path / "campaigns" / "biscuits" / "obs.csv"
obs_std_path = resources_path / "campaigns" / "biscuits" / "obs_std.csv"
obs_df = pd.read_csv(obs_path)
obs_std_df = pd.read_csv(obs_std_path)

### ###

# Instantiate the dataset and emulator classes
dataset = tl.Dataset(dataset_id)
emulator = tl.Emulator(emulator_id)

# Set up the core functions
core_functions = [
    tl.list_datasets,
    tl.list_emulators,
    tl.user_information,
    tl.versions,
    # tl.get_api_key,
    # tl.get_server_url,
    # tl.list_example_datasets,
    # tl.load_example_dataset,
    # tl.set_api_key,
    # tl.set_server_url,
]

# Package parameters in a dictionary to make them easier to loop through
dataset_method_kwargs = {
    "upload": {"df": df},
    "analyse_variance": {"columns": inputs},
    "view": {},
    "summarise": {},
    "delete": {},
}
emulator_method_kwargs = {
    "train": {
        "dataset": dataset,
        "inputs": inputs,
        "outputs": outputs,
        "params": tl.TrainParams(
            train_test_ratio=train_test_ratio,
            seed=seed,
        ),
    },
    "view": {},
    "summarise": {},
    "score": {},
    "benchmark": {},
    "predict": {"df": eval_df},
    "sample": {"df": eval_df, "num_samples": num_samples},
    "recommend": {"num_points": num_points, "acq_func": acq_func},
    "calibrate": {"df_obs": obs_df, "df_std": obs_std_df},
    "delete": {},
}

# Set up the results dataframes
core_data = pd.DataFrame()
core_data.name = "core functions"
dataset_data = pd.DataFrame()
dataset_data.name = "dataset methods"
emulator_data = pd.DataFrame()
emulator_data.name = "emulator methods"

# Time core methods
for func in core_functions:
    core_data[func.__name__] = time_function(num_iterations, func)

# Time Dataset methods
for method_name, kwargs in dataset_method_kwargs.items():
    dataset_data[f"dataset_{method_name}"] = time_method(
        num_iterations, dataset, method_name, **kwargs
    )

# Time the emulator methods
dataset.upload(df)
for method_name, kwargs in emulator_method_kwargs.items():
    emulator_data[f"emulator_{method_name}"] = time_method(
        num_iterations, emulator, method_name, **kwargs
    )
data_list = [core_data, dataset_data, emulator_data]

# Print the results if desired
if verbose:
    for df in data_list:
        print(df)

if record:
    recording_directory = create_timestamp_directory()
    record_results(data_list, recording_directory)
