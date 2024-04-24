import sys
from pprint import pprint

import dotenv

from api import use_request_model

dotenv_path = dotenv.find_dotenv(usecwd=True)
dotenv.load_dotenv(dotenv_path, override=True)

if len(sys.argv) != 4:
    print(
        f"Usage: python {sys.argv[0]} <model_id> <path/to/obs_data.csv> <path/to/obs_std_data.csv>"
    )
    exit()
model_id = sys.argv[1]
obs_csv = sys.argv[2]
obs_std_csv = sys.argv[3]
processor = "cpu"

obs_csv = open(obs_csv, "r").read()
obs_std_csv = open(obs_std_csv, "r").read()
response = use_request_model(
    model_id,
    "solve_inverse",
    data_csv=obs_csv,
    data_std_csv=obs_std_csv,
    processor=processor,
    verbose=True,
)
pprint(response)
