from pprint import pprint
import sys

import dotenv

from api import view_data_model

dotenv_path = dotenv.find_dotenv(usecwd=True)
dotenv.load_dotenv(dotenv_path, override=True)

if len(sys.argv) != 3:
    print(f"Usage: python {sys.argv[0]} <model_id> <dataset_type>")
    exit()
model_id = sys.argv[1]
dataset_type = sys.argv[2]

response = view_data_model(model_id, dataset_type=dataset_type, verbose=True)
pprint(response)
