import os

import twinlab as tl

# Dataset
# NOTE: The resources path is relative to where you call the test
# NOTE: Calling this test from python level you do two ".."
# NOTE: Calling this test from inside this file's directory you need four ".."
resources_path = os.path.join("..", "..", "resources")
training_path = os.path.join(resources_path, "datasets", "biscuits.csv")
dataset_id = "biscuits"

# Training
inputs = ["Pack price [GBP]", "Number of biscuits per pack"]
outputs = ["Number of packs sold", "Profit [GBP]"]
emulator_id = "biscuits"
campaign_path = os.path.join(resources_path, "campaigns", "biscuits", "params.json")
campaign = tl.load_params(campaign_path)

# Inference
predict_path = os.path.join(resources_path, "campaigns", "biscuits", "eval.csv")
obs_path = os.path.join(resources_path, "campaigns", "biscuits", "obs.csv")
obs_std_path = os.path.join(resources_path, "campaigns", "biscuits", "obs_std.csv")
num_samples = 5
num_points = 5
acq_func = "qExpectedImprovement"
iterations = 100
n_chains = 2
priors = [
    tl.Prior("x1", tl.distributions.Uniform(0, 12)),
    tl.Prior("x1", tl.distributions.Uniform(0, 0.5)),
    tl.Prior("x1", tl.distributions.Uniform(0, 10)),
]
num_points = 5

# General
verbose = True

# User information
tl.user_information(verbose=verbose)
tl.versions(verbose=verbose)

# Datasets
df = tl.load_dataset(training_path)
df_predict = tl.load_dataset(predict_path)
df_obs = tl.load_dataset(obs_path)
df_std = tl.load_dataset(obs_std_path)

# Check that the trainging dataset has been uploaded
tl.list_datasets(verbose=verbose)
tl.list_example_datasets(verbose=verbose)
tl.load_example_dataset(dataset_id, verbose=verbose)

# Upload dataset to the cloud and view
dataset = tl.Dataset(dataset_id)
dataset.upload(df, verbose=verbose)
dataset.view(verbose=verbose)
dataset.summarise(verbose=verbose)

# Analyse dataset
dataset.analyse_variance(columns=inputs, verbose=verbose)
dataset.analyse_variance(columns=outputs, verbose=verbose)

# Check that the emulator has been trained
tl.list_emulators(verbose=verbose)

# Set up the emulator
emulator = tl.Emulator(emulator_id)
params = tl.TrainParams(
    train_test_ratio=campaign["train_test_ratio"],
    seed=campaign["seed"],
)
process_id = emulator.train(  # Need this to return process_id
    dataset,
    campaign["inputs"],
    campaign["outputs"],
    params=params,
    wait=False,  # So that a process_id is returned
    verbose=verbose,
)
emulator.status(process_id, verbose=verbose)
emulator.train(  # Retrain the emulator normally
    dataset,
    campaign["inputs"],
    campaign["outputs"],
    params=params,
    verbose=verbose,
)

# View properties of the trained emulator
emulator.view_train_data(verbose=verbose)
emulator.view_test_data(verbose=verbose)
emulator.view(verbose=verbose)
emulator.summarise(verbose=verbose)

# Test methods of the emulator
emulator.design(priors=priors, num_points=num_points, verbose=verbose)
emulator.score(verbose=verbose)
emulator.benchmark(verbose=verbose)
emulator.predict(df_predict, verbose=verbose)
emulator.sample(df_predict, num_samples, verbose=verbose)
emulator.recommend(num_points, acq_func, verbose=verbose)
params = tl.CalibrateParams(iterations=100, seed=0)
emulator.calibrate(df_obs, df_std, params=params, verbose=verbose)

# View processes of the methods run on the emulator
emulator.list_processes(verbose=verbose)

# Delete dataset and emulator from the cloud
emulator.delete(verbose=verbose)
dataset.delete(verbose=verbose)
