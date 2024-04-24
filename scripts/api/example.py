import os
import time
from pprint import pprint as pprint_original

import dotenv

import api

# Load environment variables
dotenv_path = dotenv.find_dotenv(usecwd=True)
dotenv.load_dotenv(dotenv_path, override=True)

# General
# NOTE: The resources path is relative to where you call the test
# NOTE: Calling this test from python level you do two ".."
# NOTE: Calling this test from inside this file's directory you need four ".."
resources_path = os.path.join("..", "..", "resources")
wait_time = 1.0
processor = "cpu"

# Dataset
dataset_path = os.path.join(resources_path, "datasets", "gardening.csv")
dataset_id = "gardening"

# Method: analyse_dataset
analysis_columns = "Sunlight [hours/day],Water [times/week]"

# Initial design
priors = [
    '{"name": "x1", "distribution": {"method": "uniform", "distribution_params": {"max": 12, "min": 0}}}',
    '{"name": "x2", "distribution": {"method": "uniform", "distribution_params": {"max": 0.5, "min": 0}}}',
    '{"name": "x3", "distribution": {"method": "uniform", "distribution_params": {"max": 10, "min": 0}}}',
]
sampling_method = {
    "method": "latin_hypercube",
    "sampling_params": {"scramble": True, "optimization": "random-cd"},
}
num_design = 10

# Model
params_path = os.path.join(resources_path, "campaigns", "gardening", "params.json")
model_id = "gardening"

# Method: predict
predict_path = os.path.join(resources_path, "campaigns", "gardening", "eval.csv")

# Method: sample
sample_path = predict_path
num_samples = 5

# Method: get_candidate_points
acq_func = "qNIPV"
num_points = 1  # NOTE: Timeout if num_points > 1 with synchronous endpoint

# Method: solve_inverse
obs_path = os.path.join(resources_path, "campaigns", "gardening", "obs.csv")
obs_std_path = os.path.join(resources_path, "campaigns", "gardening", "obs_std.csv")
num_iterations = 10


# Redefine pprint
def pprint(msg):
    return pprint_original(msg, compact=True, sort_dicts=False)


response = api.get_user(verbose=True)
pprint(response)

response = api.get_versions(verbose=True)
pprint(response)

response = api.list_datasets(verbose=True)
pprint(response)

response = api.list_example_datasets(verbose=True)
pprint(response)

response = api.load_example_dataset(dataset_id, verbose=True)
pprint(response)

data_csv = open(dataset_path, "r").read()
response = api.upload_dataset(dataset_id, data_csv, verbose=True)
pprint(response)

response = api.view_dataset(dataset_id, verbose=True)
pprint(response)

response = api.summarise_dataset(dataset_id, verbose=True)
pprint(response)

response = api.analyse_dataset(dataset_id, analysis_columns, verbose=True)
pprint(response)

response = api.list_models(verbose=True)
pprint(response)

response = api.get_initial_design(
    priors,
    sampling_method,
    num_design,
    verbose=True,
)
pprint(response)

# Train model
parameters_json = open(params_path, "r").read()
response = api.train_model(model_id, parameters_json, processor, verbose=True)
pprint(response)
complete = False
while not complete:  # Wait for job to complete
    response = api.get_status_model(model_id, verbose=True)
    pprint(response)
    _, body = response
    complete = body["job_complete"]
    time.sleep(wait_time)

# Train model v2
response = api.train_request_model(model_id, parameters_json, processor, verbose=True)
pprint(response)
_, body = response
process_id = body["process_id"]
status = 202
while status == 202:  # Wait for job to complete
    response = api.train_response_model(model_id, process_id, verbose=True)
    pprint(response)
    status, _ = response
    time.sleep(wait_time)

response = api.view_data_model(model_id, dataset_type="train", verbose=True)
pprint(response)

response = api.view_data_model(model_id, dataset_type="test", verbose=True)
pprint(response)

response = api.view_model(model_id, verbose=True)
pprint(response)

response = api.summarise_model(model_id, verbose=True)
pprint(response)

# Score and get_calibration_curve
for method in ["score", "get_calibration_curve"]:
    response = api.use_model(
        model_id,
        method=method,
        processor=processor,
        verbose=True,
    )
    pprint(response)

# Setup for methods
predict_csv = open(predict_path, "r").read()
sample_csv = open(sample_path, "r").read()
obs_csv = open(obs_path, "r").read()
obs_std_csv = open(obs_std_path, "r").read()

# Method and kwargs key/values
use_model_dict = {
    # "score": {}, # TODO: Does not seem to work asynchonously
    # "get_calibration_curve": {}, # TODO: Does not seem to work asynchonously
    "predict": {"data_csv": predict_csv},
    "sample": {"data_csv": sample_csv, "num_samples": num_samples},
    "get_candidate_points": {"acq_func": acq_func, "num_points": num_points},
    "solve_inverse": {
        "data_csv": obs_csv,
        "data_std_csv": obs_std_csv,
        "kwargs": {"iterations": num_iterations},
    },
}

# Loop over methods/kwargs
for method, kwargs in use_model_dict.items():

    # Synchronous
    response = api.use_model(
        model_id,
        method=method,
        processor=processor,
        verbose=True,
        **kwargs,
    )
    pprint(response)

    # Asynchronous
    response = api.use_request_model(
        model_id,
        method=method,
        processor=processor,
        verbose=True,
        **kwargs,
    )
    pprint(response)
    _, body = response
    process_id = body["process_id"]
    status = 202
    while status == 202:
        response = api.use_response_model(model_id, method, process_id, verbose=True)
        pprint(response)
        status, _ = response
        time.sleep(wait_time)

response = api.list_processes_model(model_id, verbose=True)
pprint(response)

response = api.delete_model(model_id, verbose=True)
pprint(response)

response = api.delete_dataset(dataset_id, verbose=True)
pprint(response)
