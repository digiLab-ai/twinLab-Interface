import sys
import twinlab as tl

if len(sys.argv) != 3:
    print(f"Usage: python {sys.argv[0]} <emulator_id> <path/to/dataset.csv>")
    exit()

emulator_id = sys.argv[1]
filepath = sys.argv[2]

emulator = tl.Emulator(id=emulator_id)
df = tl.load_dataset(filepath)
emulator.predict(df, verbose=True)
