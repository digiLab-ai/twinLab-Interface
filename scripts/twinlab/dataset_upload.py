import sys

import twinlab as tl


if len(sys.argv) != 3:
    print(f"Usage: python {sys.argv[0]} <dataset_id> <path/to/dataset.csv>")
    exit()

dataset_id = sys.argv[1]
filepath = sys.argv[2]

df = tl.load_dataset(filepath)
tl.Dataset(dataset_id).upload(df, verbose=True)
