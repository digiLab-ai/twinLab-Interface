import sys

import twinlab as tl


if len(sys.argv) != 2:
    print(f"Usage: python {sys.argv[0]} <dataset_id> <columns>")
    exit()

columns = ["Pack price [GBP]", "Number of biscuits per pack"]
dataset_id = sys.argv[1]

response = tl.Dataset(dataset_id).analyse_variance(columns)
print(response)
