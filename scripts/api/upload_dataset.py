import sys
from pprint import pprint

import dotenv

from api import upload_dataset

dotenv_path = dotenv.find_dotenv(usecwd=True)
dotenv.load_dotenv(dotenv_path, override=True)

if len(sys.argv) != 3:
    print(f"Usage: python {sys.argv[0]} <dataset_id> <path/to/dataset.csv>")
    exit()
dataset_id = sys.argv[1]
filepath_csv = sys.argv[2]

data_csv = open(filepath_csv, "r").read()
response = upload_dataset(dataset_id, data_csv, verbose=True)
pprint(response)
