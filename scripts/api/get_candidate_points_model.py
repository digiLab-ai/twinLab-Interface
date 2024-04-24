import sys
from pprint import pprint

import dotenv

from api import use_model

dotenv_path = dotenv.find_dotenv(usecwd=True)
dotenv.load_dotenv(dotenv_path, override=True)

if len(sys.argv) != 4:
    print(f"Usage: python {sys.argv[0]} <model_id> <acq_func> <number_of_points>")
    exit()
model_id = sys.argv[1]
acq_func = sys.argv[2]
num_points = int(sys.argv[3])
processor = "cpu"

response = use_model(
    model_id,
    "get_candidate_points",
    processor=processor,
    verbose=True,
    acq_func=acq_func,
    num_points=num_points,
)
pprint(response)
