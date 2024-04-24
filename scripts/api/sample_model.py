import sys
from pprint import pprint

import dotenv

from api import use_model

dotenv_path = dotenv.find_dotenv(usecwd=True)
dotenv.load_dotenv(dotenv_path, override=True)

if len(sys.argv) != 4:
    print(f"Usage: python {sys.argv[0]} <model_id> <path/to/inputs.csv> <num_samples>")
    exit()
model_id = sys.argv[1]
filepath_csv = sys.argv[2]
num_samples = int(sys.argv[3])
processor = "cpu"

eval_csv = open(filepath_csv, "r").read()
response = use_model(
    model_id,
    "sample",
    data_csv=eval_csv,
    processor=processor,
    verbose=True,
    num_samples=num_samples,
)
pprint(response)
