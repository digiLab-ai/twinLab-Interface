import sys
from pprint import pprint

import dotenv

from api import view_model

dotenv_path = dotenv.find_dotenv(usecwd=True)
dotenv.load_dotenv(dotenv_path, override=True)

if len(sys.argv) != 2:
    print(f"Usage: python {sys.argv[0]} <model_id>")
    exit()
model_id = sys.argv[1]

response = view_model(model_id, verbose=True)
pprint(response)
