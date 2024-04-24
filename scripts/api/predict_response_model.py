import sys
from pprint import pprint

import dotenv

from api import use_response_model

dotenv_path = dotenv.find_dotenv(usecwd=True)
dotenv.load_dotenv(dotenv_path, override=True)

if len(sys.argv) != 3:
    print(f"Usage: python {sys.argv[0]} <model_id> <process_id>")
    exit()
model_id = sys.argv[1]
process_id = sys.argv[2]

response = use_response_model(model_id, "predict", process_id, verbose=True)
pprint(response)
