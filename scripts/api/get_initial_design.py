from pprint import pprint
import sys

import dotenv

from api import get_initial_design

dotenv_path = dotenv.find_dotenv(usecwd=True)
dotenv.load_dotenv(dotenv_path, override=True)

priors = [
    '{"name": "x1", "distribution": {"method": "uniform", "distribution_params": {"max": 12, "min": 0}}}',
    '{"name": "x2", "distribution": {"method": "uniform", "distribution_params": {"max": 0.5, "min": 0}}}',
    '{"name": "x3", "distribution": {"method": "uniform", "distribution_params": {"max": 10, "min": 0}}}',
]
sampling_method = {
    "method": "latin_hypercube",
    "sampling_params": {"scramble": True, "optimization": "random-cd"},
}

if len(sys.argv) != 2:
    print(f"Usage: python {sys.argv[0]} <num_points>")
    exit()

num_points = int(sys.argv[1])

response = get_initial_design(
    priors,
    sampling_method,
    num_points,
    verbose=True,
)
pprint(response)
