import sys
from pprint import pprint

import dotenv

from api import train_request_model

dotenv_path = dotenv.find_dotenv(usecwd=True)
dotenv.load_dotenv(dotenv_path, override=True)

if len(sys.argv) != 3:
    print(f"Usage: python {sys.argv[0]} <model_id> <path/to/parameters.json>")
    exit()
model_id = sys.argv[1]
filepath_json = sys.argv[2]
processor = "cpu"

parameters_json = open(filepath_json, "r").read()
response = train_request_model(model_id, parameters_json, processor, verbose=True)
pprint(response)
