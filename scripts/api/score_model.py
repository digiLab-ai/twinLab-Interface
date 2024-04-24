import sys
from pprint import pprint

import dotenv

from api import use_model

dotenv_path = dotenv.find_dotenv(usecwd=True)
dotenv.load_dotenv(dotenv_path, override=True)

if len(sys.argv) != 4:
    print(f"Usage: python {sys.argv[0]} <model_id> <metric> <combined_score>")
    exit()
model_id = sys.argv[1]
metric = sys.argv[2]
combined_score = eval(sys.argv[3])
processor = "cpu"

response = use_model(
    model_id,
    "score",
    metric=metric,
    combined_score=combined_score,
    processor=processor,
    verbose=True,
)
pprint(response)
