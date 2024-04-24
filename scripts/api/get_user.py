from pprint import pprint

import dotenv

from api import get_user

dotenv_path = dotenv.find_dotenv(usecwd=True)
dotenv.load_dotenv(dotenv_path, override=True)

response = get_user(verbose=True)
pprint(response)
