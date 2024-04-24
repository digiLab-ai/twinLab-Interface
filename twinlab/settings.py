# Standard imports
import os

# Third-party imports
import dotenv

# Project imports
from ._version import __version__


# Parameters
# TODO: Move these into a settings.json?
DEFAULT_TWINLAB_URL = "https://twinlab.digilab.co.uk"
CHECK_DATASETS = True  # Check datasets are sensible before uploading
PARAMS_COERCION = {  # Convert parameter names in params dict
    "test_train_ratio": "train_test_ratio",  # Common mistake
    "filename": "dataset_id",  # Support old name
    "filename_std": "dataset_std_id",  # Support old name
    "filename_stdv": "dataset_std_id",
    "filename_stdev": "dataset_std_id",
    "dataset": "dataset_id",  # Support old name
    "dataset_std": "dataset_std_id",  # Support old name
    "dataset_stdv": "dataset_std_id",
    "dataset_stdev": "dataset_std_id",
    "functional_input": "decompose_inputs",
    "functional_output": "decompose_outputs",
    "function_input": "decompose_inputs",
    "function_output": "decompose_outputs",
}

# Load environment variables from .env, if it exists
# NOTE: Should search from current directory upwards
dotenv_path = dotenv.find_dotenv(usecwd=True)
dotenv.load_dotenv(dotenv_path, override=True)

# Set defaults if not set
if not os.getenv("TWINLAB_URL"):
    os.environ["TWINLAB_URL"] = DEFAULT_TWINLAB_URL
if not os.getenv("TWINLAB_API_KEY"):
    os.environ["TWINLAB_API_KEY"] = "None"

# Intro message
print()
print(f"          ====== TwinLab Client Initialisation ======")
print(f"          Version     : {__version__}")
print(f"          Server      : {os.getenv('TWINLAB_URL')}")
if dotenv_path:
    print(f"          Environment : {dotenv_path}")
print()
