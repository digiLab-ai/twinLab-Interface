import sys
from pprint import pprint

import dotenv

from api import use_model

dotenv_path = dotenv.find_dotenv(usecwd=True)
dotenv.load_dotenv(dotenv_path, override=True)

if len(sys.argv) != 3:
    print(f"Usage: python {sys.argv[0]} <model_id> <benchmark_type>")
    exit()

model_id = sys.argv[1]
benchmark_type = sys.argv[2]
processor = "cpu"

response = use_model(
    model_id,
    "get_calibration_curve",
    type=benchmark_type,
    processor=processor,
    verbose=True,
)
pprint(response)
