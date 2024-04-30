from pprint import pprint
import sys

import dotenv

from api import load_example_dataset

dotenv_path = dotenv.find_dotenv(usecwd=True)
dotenv.load_dotenv(dotenv_path, override=True)

if len(sys.argv) != 2:
    print(f"Usage: python {sys.argv[0]} <dataset_id>")
    exit()
dataset_id = sys.argv[1]

response = load_example_dataset(dataset_id, verbose=True)
pprint(response)