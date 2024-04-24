import sys
import twinlab as tl

if len(sys.argv) != 2:
    print(f"Usage: python {sys.argv[0]} <dataset_id>")
    exit()
dataset_id = sys.argv[1]

tl.load_example_dataset(dataset_id, verbose=True)
