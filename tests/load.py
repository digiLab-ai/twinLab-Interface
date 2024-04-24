import concurrent.futures
import sys

import pandas as pd
import twinlab as tl

if len(sys.argv) != 3:
    print(f"Usage: python {sys.argv[0]} <num_jobs> <processor: cpu/gpu>")
    exit()

# Create and upload a dataset
df = pd.DataFrame({"X": [1, 2, 3, 4], "y": [1, 4, 9, 16]})
dataset = tl.Dataset(f"queuing_dataset")
dataset.upload(df)

# Define the number of emulators and the processor
num_emulators = int(sys.argv[1])
processor = sys.argv[2]

emulator_dict = {}

for i in range(num_emulators):

    # Define a method that creates an emulator and trains it
    def method(i):
        emulator = tl.Emulator(str(i))
        emulator.train(dataset, ["X"], ["y"], verbose=True, processor=processor)
        emulator.delete()

    # Store the method in a dictionary
    emulator_dict[f"method_{i}"] = method

# Create a ThreadPoolExecutor with the desired number of worker threads
with concurrent.futures.ThreadPoolExecutor(max_workers=num_emulators) as executor:

    # Submit the methods to the executor
    for i in range(num_emulators):
        futures = [executor.submit(emulator_dict[f"method_{i}"], i)]

    # Wait for all futures to complete
    concurrent.futures.wait(futures)
