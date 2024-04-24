from pprint import pprint
import sys

import dotenv

from api import list_processes_model

dotenv_path = dotenv.find_dotenv(usecwd=True)
dotenv.load_dotenv(dotenv_path, override=True)

if len(sys.argv) != 2:
    print(f"Usage: python {sys.argv[0]} <model_id>")
    exit()
model_id = sys.argv[1]

response = list_processes_model(model_id, verbose=True)
pprint(response)
