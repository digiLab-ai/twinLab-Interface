import sys
import twinlab as tl

if len(sys.argv) != 4:
    print(
        f"Usage: python {sys.argv[0]} <emulator_id> <path/to/dataset.csv> <number_of_samples>"
    )
    exit()

emulator_id = sys.argv[1]
filepath = sys.argv[2]
n_samples = int(sys.argv[3])

emulator = tl.Emulator(id=emulator_id)
df = tl.load_dataset(filepath)
emulator.sample(df, n_samples, verbose=True)
