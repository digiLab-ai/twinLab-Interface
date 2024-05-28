# Version
from ._version import __version__

# Campaign functions
# NOTE: Deprecated (v1)
# Dataset functions
# NOTE: Deprecated (v1)
# General functions
# NOTE: Deprecated (v1)
from .client import (
    active_learn_campaign,
    delete_campaign,
    delete_dataset,
    get_api_key,
    get_calibration_curve_campaign,
    get_server_url,
    get_user_information,
    get_versions,
    list_campaigns,
    list_datasets,
    optimise_campaign,
    predict_campaign,
    query_campaign,
    query_dataset,
    sample_campaign,
    score_campaign,
    set_api_key,
    set_server_url,
    solve_inverse_campaign,
    train_campaign,
    upload_dataset,
    view_campaign,
    view_dataset,
)

# General functions
from .core import (
    get_api_key,
    get_server_url,
    list_datasets,
    list_emulators,
    list_example_datasets,
    load_example_dataset,
    set_api_key,
    set_server_url,
    user_information,
    versions,
)
from .dataset import Dataset

# Distribution class
from .distributions import Distribution
from .emulator import Emulator
from .helper import get_sample, join_samples, load_dataset, load_params
from .params import (
    AcqFuncParams,
    BenchmarkParams,
    CalibrateParams,
    DesignParams,
    EstimatorParams,
    ModelSelectionParams,
    MaximizeParams,
    OptimiserParams,
    PredictParams,
    RecommendParams,
    SampleParams,
    ScoreParams,
    TrainParams,
)

# Prior class
from .prior import Prior

# Sampling methods
from .sampling import Sampling
