import sys

import numpy as np
import pandas as pd
import twinlab as tl

# Get arguments from the command line
# NOTE: the emulator and dataset you specify will not be pulled from your cloud account. Dummies are used here
# NOTE: this is an example script with a hard-coded example
if len(sys.argv) != 6:
    print(
        f"Usage: python {sys.argv[0]} <emulator_id> <dataset_id> <num_loops> <num_points_per_loop> <acq_function>"
    )
    exit()
emulator_id = sys.argv[1]
dataset_id = sys.argv[2]
num_loops = int(sys.argv[3])
num_points_per_loop = int(sys.argv[4])
acq_function = sys.argv[5]


# Defining the simulation function
def target_function(x, y):
    # Branin function
    a = 1
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)
    return a * (y - b * x**2 + c * x - r) ** 2 + s * (1 - t) * np.cos(x) + s


# Get the function into the correct format for Emulator.learn()
combined_function = lambda X: target_function(X[:, 0], X[:, 1])

# Generate training data
xmin, xmax, ymin, ymax = -5, 10, 0, 15
n_train = 10
err = 0.1
x_train = np.random.uniform(-5, xmax, n_train)
y_train = np.random.uniform(ymin, ymax, n_train)
z_train = np.random.normal(target_function(x_train, y_train), err, n_train)
df = pd.DataFrame(
    {"x": x_train.flatten(), "y": y_train.flatten(), "z": z_train.flatten()}
)

# Initialize the emulator and dataset
inputs = ["x", "y"]
outputs = ["z"]
dataset = tl.Dataset(dataset_id)
dataset.upload(df)
emulator = tl.Emulator(emulator_id)

# Training parameters
params = tl.TrainParams(train_test_ratio=1.0)

# Call the learn function
emulator.learn(
    dataset=dataset,
    inputs=inputs,
    outputs=outputs,
    num_loops=num_loops,
    num_points_per_loop=num_points_per_loop,
    acq_func=acq_function,
    simulation=combined_function,
    train_params=params,
    verbose=True,
)
