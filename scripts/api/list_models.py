from pprint import pprint

import dotenv

from api import list_models

dotenv_path = dotenv.find_dotenv(usecwd=True)
dotenv.load_dotenv(dotenv_path, override=True)

response = list_models(verbose=True)
pprint(response)
