from pprint import pprint

import dotenv

from api import analyse_dataset

dotenv_path = dotenv.find_dotenv(usecwd=True)
dotenv.load_dotenv(dotenv_path, override=True)

response = analyse_dataset(
    "biscuits",
    "Pack price [GBP],Number of biscuits per pack",
)
pprint(response)
