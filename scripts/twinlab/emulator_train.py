import sys
import twinlab as tl

if len(sys.argv) != 3:
    print(f"Usage: python {sys.argv[0]} <emulator_id> <training_params>")
    exit()

emulator_id = sys.argv[1]
training_params = sys.argv[2]
params = tl.load_params(training_params)

# Extract the dataset_id, inputs, and outputs from the parameters
dataset_id = params.pop("dataset_id")
inputs = params.pop("inputs")
outputs = params.pop("outputs")

# Remove these keys from the parameters if they exist
params.pop("decompose_inputs", None)
params.pop("decompose_outputs", None)

emulator = tl.Emulator(emulator_id)
params = tl.TrainParams(**params)
emulator.train(tl.Dataset(dataset_id), inputs, outputs, params=params, verbose=True)
