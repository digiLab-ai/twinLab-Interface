import sys
from pprint import pprint

import dotenv

from api import use_request_model

dotenv_path = dotenv.find_dotenv(usecwd=True)
dotenv.load_dotenv(dotenv_path, override=True)

if len(sys.argv) != 3:
    print(f"Usage: python {sys.argv[0]} <model_id> <path/to/inputs.csv>")
    exit()
model_id = sys.argv[1]
filepath_csv = sys.argv[2]
processor = "cpu"

eval_csv = open(filepath_csv, "r").read()
response = use_request_model(
    model_id, "predict", data_csv=eval_csv, processor=processor, verbose=True
)
pprint(response)
