# Standard imports
import os
from enum import Enum

# Third-party imports
import dotenv

# Project imports
from ._version import __version__
import warnings


# Possible states of a twinLab job
class ValidStatus(Enum):
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILURE = "failure"


# Not a good name?
class ValidExportFormats(Enum):
    TORCHSCRIPT = "torchscript"

    @staticmethod
    def list():
        return [format.value for format in ValidExportFormats]


# FMU Valid Export Formats


class ValidFMUTypes(Enum):
    MODEL_EXCHANGE = "model-exchange"
    # CO_SIMULATION = "CoSimulation"

    @staticmethod
    def list():
        return [fmu_type.value for fmu_type in ValidFMUTypes]


class ValidFMUOS(Enum):
    WINDOWS = "win64"
    # LINUX = "linux"
    # MAC = "darwin64"

    @staticmethod
    def list():
        return [os.value for os in ValidFMUOS]


# Parameters
DEFAULT_TWINLAB_URL = "https://twinlab.digilab.co.uk/v3"
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
# NOTE: Should search from current directory outwards
dotenv_path = dotenv.find_dotenv(usecwd=True)
# Try to load the .env file.
try:
    dotenv.load_dotenv(dotenv_path)
except UnicodeDecodeError as e:
    warnings.warn("Failed to load environment variables from .env file.")
    print(".env location:", dotenv_path)

# Set defaults if not set
if not os.getenv("TWINLAB_URL"):
    os.environ["TWINLAB_URL"] = DEFAULT_TWINLAB_URL

# Intro message
print()
print(f"          ====== TwinLab Client Initialisation ======")
print(f"          Version     : {__version__}")
if os.getenv("TWINLAB_USER"):
    print(f"          User        : {os.getenv('TWINLAB_USER')}")
print(f"          Server      : {os.getenv('TWINLAB_URL')}")
if dotenv_path:
    print(f"          Environment : {dotenv_path}")
print()
