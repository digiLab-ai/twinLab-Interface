# Standard imports
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Tuple, Union

import pandas as pd

# Third-party imports
from deprecated import deprecated
from typeguard import typechecked

from .dataset import Dataset
from .sampling import LatinHypercube, Sampling

# Project imports
from .utils import remove_none_values

ALLOWED_ESTIMATORS = [
    "single_task_gp",
    "fixed_noise_gp",
    "heteroskedastic_gp",
    "variational_gp",
    "multi_fidelity_gp",
    "fixed_noise_multi_fidelity_gp",
    "mixed_single_task_gp",
]

ESTIMATORS_WITHOUT_COVAR_MODULE = [
    "heteroskedastic_gp",
    "multi_fidelity_gp",
    "fixed_noise_multi_fidelity_gp",
    "mixed_single_task_gp",
]

DEFAULT_SEED = 42


def _check_estimator_type(estimator_type: str):
    # NOTE: This is necessary because the library defaults to `single_task_gp` if the estimator_type is not provided or wrong!
    if estimator_type not in ALLOWED_ESTIMATORS:
        raise ValueError(
            f"Estimator type {estimator_type} not recognised. Please choose from {ALLOWED_ESTIMATORS}."
        )


def _check_estimator_type_and_covar_module(
    estimator_type: str, covar_module: Optional[str]
):
    # Logic to avoid sending the covar_module arg to the library for estimators that do not take it as an argument
    # TODO: This should be moved to the library level?
    if (estimator_type in ESTIMATORS_WITHOUT_COVAR_MODULE) and (
        covar_module is not None
    ):
        raise ValueError(
            f"The parameter covar_module cannot be set for the estimator {estimator_type}."
        )


# Logic to convert a DataFrame to a dictionary to allow serialisation
def _convert_dataframe_to_dict(dictionary: dict, value: str):
    if value in dictionary:
        if type(dictionary[value]) is pd.DataFrame:
            dictionary[value] = dictionary[value].to_dict()
        return dictionary[value]


@typechecked
class Params(ABC):
    """Abstract base class for all parameter classes"""

    @abstractmethod
    def unpack_parameters(self):
        pass


@typechecked
class EstimatorParams(Params):
    """Parameter configuration for the Gaussian Process emulator (estimator).

    Attributes:
        detrend (bool, optional): Should the linear trend in the data be removed (detrended) before training the emulator?
            The defaults is ``False``.
        covar_module (Union[str, None], optional): Specifies the functions that build up the kernel (covariance matrix) of the Gaussian Process.
            The default is ``None``, which means the library will use a default kernel, which is a scaled Matern 5/2.
            This can be chosen from a list of possible kernels:

            - ``"LIN"``: Linear.
            - ``"M12"``: Matern 1/2. A standard kernel for modelling data with a smooth trend
            - ``"M32"``: Matern 3/2. A standard kernel for modelling data with a smooth trend.
            - ``"M52"``: Matern 5/2. A standard kernel for modelling data with a smooth trend.
            - ``"PER"``: Periodic. Good for modelling data that has a periodic structure.
            - ``"RBF"``: Radial Basis Function. A standard kernel for modelling data with a smooth trend.
              A good default choice that can model smooth functions.
            - ``"RQF"``: Rational Quadratic Function.

            Kernels can also be composed by using combinations of the ``"+"`` (addative) and ``"*"`` (multiplicative) operators.
            For example, ``covar_module = "(M52*PER)+RQF"`` is valid.
        estimator_type (str, optional): Specifies the type of Gaussian process to use for the emulator.
            The default is ``"single_task_gp"``, but the value can be chosen from the following list:

            - ``"single_task_gp"``: The standard Gaussian Process, which learns a mean, covariance, and noise level.
            - ``"fixed_noise_gp"``: A Gaussian Process with fixed noise, which is specified by the user.
              Particularly useful for modelling noise-free simulated data where the noise can be set to zero manually.
            - ``"heteroskedastic_gp"``: A Gaussian Process with fixed noise that is allowed to vary with the input.
              The noise is specified by the user, and is also learned by the Process.
            - ``"variational_gp"``: An approximate Gaussian Process that is more efficient to train with large datasets.
            - ``"mixed_single_task_gp"``: A Gaussian Process that works with a mix of continuous and categorical or discrete input data.
            - ``"multi_fidelity_gp"``:  A Gaussian Process that works with input data that has multiple levels of fidelity.
              For example, combined data from both a high- and low-resolution simulation.
            - ``"fixed_noise_multi_fidelity_gp"``: A Gaussian Process that works with input data that has multiple levels of fidelity and fixed noise.
    """

    def __init__(
        self,
        detrend: bool = False,
        covar_module: Optional[str] = None,
        estimator_type: str = "single_task_gp",
    ):
        self.detrend = detrend
        self.covar_module = covar_module
        self.estimator_type = estimator_type
        _check_estimator_type(self.estimator_type)
        _check_estimator_type_and_covar_module(self.estimator_type, self.covar_module)

    def unpack_parameters(self):
        params = {
            "detrend": self.detrend,
            "covar_module": self.covar_module,
            "estimator_type": self.estimator_type,
        }
        params = remove_none_values(params)
        return params


@typechecked
class ModelSelectionParams(Params):
    """Parameter configuration for the Bayesian model selection process.

    Attributes:
        seed (Union[int, None], optional): Specifies the seed for the random number generator for every trial of the model selection process.
            Setting to an integer is necessary for reproducible results.
            The default value is ``None``, which means the seed is randomly generated each time.
        evaluation_metric (str, optional): Specifies the evaluation metric used to score different configuration during the model selection process.
            Can be either:

            - ``"MSLL"``: Mean squared log loss.
            - ``"BIC"``: Bayesian information criterion.

            The default is ``"MSLL"``.
        val_ratio (float, optional): Specifies the percentage of random validation data allocated to to compute the ``"BIC"`` metric.
            The default is ``0.2``.
        base_kernels (Union[str, Set[str]], optional): Specifies the set of individual kernels to use for compositional kernel search.
            Can be:

            - ``"all"``: The complete set of available kernels: ``{"LIN", "M12", "M32", "M52", "PER", "RBF", "RQF"}``.
            - ``"restricted"``: The restricted set of kernels: ``{"LIN", "M32", "M52", "PER", "RBF"}``.
            - A set of strings corresponding to the individual kernels to use for kernel selection,
              for example ``{"RBF", "PER"}``.

            The default is `"restricted"`.
        depth (int, optional):
            Specifies the number of base kernels allowed to be combined in the compositional kernel search.
            For example, a ``depth=3`` search means the resulting kernel may be composed from up-to three base kernels,
            so examples of allowed kernel combinations would be ``"(LIN+PER)*RBF"`` or ``"(M12*RBF)+RQF"``.
            The default value is ``1``, which simply compares all kernel functions individually.
        beam (Union[int, None], optional): Specifies the beam width of the compositional kernel search algorithm.
            This uses a beam search algorithm to find the best kernel combination.
            This is a heuristic search algorithm that explores a graph by expanding the most promising nodes in a limited set.
            A ``beam=1`` search is exhaustive, which is algorithmically 'greedy'.
            ``beam=None`` instead performs a breadth-first search.
            ``beam > 1`` performs a beam search with the specified beam value.
            The default value is ``None``.
    """

    # TODO: This docstring needs to be massively improved!

    def __init__(
        self,
        seed: Optional[int] = None,
        evaluation_metric: str = "MSLL",
        val_ratio: float = 0.2,
        base_kernels: Union[str, Set[str]] = "restricted",
        depth: int = 1,
        beam: Optional[int] = None,
    ):
        self.seed = seed
        self.evaluation_metric = evaluation_metric
        self.val_ratio = val_ratio
        self.base_kernels = base_kernels
        self.depth = depth
        self.beam = beam

    def unpack_parameters(self):
        params = {
            "seed": self.seed,
            "evaluation_metric": self.evaluation_metric,
            "val_ratio": self.val_ratio,
            "base_kernels": self.base_kernels,
            "depth": self.depth,
            "beam": self.beam,
        }
        params = remove_none_values(params)
        return params


@typechecked
class TrainParams(Params):
    """Parameter configuration for training an emulator.

    This includes parameters that pertain directly to the training of the model,
    such as the ratio of training to testing data,
    as well as parameters that pertain to the setup of the model such as the number of dimensions to retain after decomposition.

    Attributes:
        estimator (str, optional): The type of estimator (emulator) to be trained.
            Currently only "gaussian_process_regression" is supported, which is the default value.
        estimator_params (EstimatorParams, optional): The parameters for the Gaussian Process emulator.
        input_retained_dimensions (Union[int, None], optional): The number of input dimensions to retain after applying dimensional reduction.
            Setting this to an integer cannot be done at the same time as specifying the ``input_explained_variance``.
            The default value is ``None``, which means that dimensional reduction is not applied to the input unless ``input_explained_variance`` is specified.
        input_explained_variance (Union[float, None], optional): Specifies what fraction of the variance of the input data is retained after applying dimensional reduction.
            This must be a number between 0 and 1.
            This cannot be specified at the same time as ``input_retained_dimensions``.
            The default value is ``None``, which means that dimensional reduction is not applied to the input unless ``input_retained_dimensions`` is specified.
        output_retained_dimensions (Union[int, None], optional): The number of output dimensions to retain after applying dimensional reduction.
            Setting this to an integer cannot be done at the same time as specifying the ``output_explained_variance``.
            The default value is ``None``, which means that dimensional reduction is not applied to the output unless ``output_explained_variance`` is specified.
        output_explained_variance (Union[float, None], optional): Specifies what fraction of the variance of the output data is retained after applying dimensional reduction.
            This must be a number between 0 and 1.
            This cannot be specified at the same time as ``output_retained_dimensions``.
            The default value is ``None``, which means that dimensional reduction is not applied to the output unless ``output_retained_dimensions`` is specified.
        fidelity (Union[str, None], optional): Name of the column in the dataset corresponding to the fidelity parameter if a multi-fidelity model is being trained.
            The default value is ``None``, whereby fidelity information is provided. Fidelity refers to the degree an emulator is able to reproduce the behaviour of the simulated data.
        train_test_ratio (Union[float, None], optional): Specifies the fraction of training samples in the dataset.
            This must be a number beteen 0 and 1.
            The default value is 1, which means that all of the provided data is used for training.
            This is good to make the most out of a dataset, but means that it will not be possible to score or benchmark the performance of an emulator.
        dataset_std (Union[Dataset, None], optional): A twinLab dataset object that contains the standard deviation of the training data.
            This is necessary when training a heteroskedastic or fixed noise Gaussian Process.
        model_selection (bool, optional): Whether to run Bayesian model selection, a form of automatic machine learning.
            The default value is ``False``, which simply trains the specified emulator, rather than iterating over them.
        model_selection_params (ModelSelectionParams, optional): The parameters for model selection, if it is being used.
        shuffle (bool, optional): Whether to randomly shuffle the training data before splitting it into training and testing sets.
            The default value is ``True``. Please be particularly careful while using this parameter with time-series data.
        seed (Union[int, None], optional): The seed used to initialise the random number generators for reproducibility.
            Setting to an integer is necessary for reproducible results.
            The default value is ``42``, but it can be set to ``None`` to randomly generate the seed each time.
            Be aware that the seed is used in the training process, so if the seed is set to ``None`` the training process will not be reproducible.

    """

    def __init__(
        self,
        estimator: str = "gaussian_process_regression",
        estimator_params: EstimatorParams = EstimatorParams(),
        input_explained_variance: Optional[float] = None,
        input_retained_dimensions: Optional[int] = None,
        output_explained_variance: Optional[float] = None,
        output_retained_dimensions: Optional[int] = None,
        fidelity: Optional[str] = None,
        dataset_std: Optional[Dataset] = None,
        train_test_ratio: float = 1.0,
        model_selection: bool = False,
        model_selection_params: ModelSelectionParams = ModelSelectionParams(),
        shuffle: bool = True,
        seed: Optional[int] = DEFAULT_SEED,
    ):
        # Parameters that will be passed to the emulator on construction
        self.fidelity = fidelity
        self.estimator = estimator
        self.estimator_params = estimator_params
        self.input_explained_variance = input_explained_variance
        self.input_retained_dimensions = input_retained_dimensions
        self.output_explained_variance = output_explained_variance
        self.output_retained_dimensions = output_retained_dimensions

        # Parameters that will be passed to the emulator.fit() method
        self.dataset_std = dataset_std
        self.train_test_ratio = train_test_ratio
        self.model_selection = model_selection
        self.model_selection_params = model_selection_params
        self.shuffle = shuffle
        self.seed = seed

        # Figure out whether or not to set decompose booleans
        def decompose(
            explained_variance: Optional[float],
            retained_dimensions: Optional[int],
        ) -> bool:
            if explained_variance is not None and retained_dimensions is not None:
                raise ValueError(
                    "Explained variance and retained dimensions cannot be set simultaneously. Please choose one."
                )
            elif explained_variance is None and retained_dimensions is None:
                return False
            else:
                return True

        self.decompose_inputs = decompose(
            self.input_explained_variance, self.input_retained_dimensions
        )
        self.decompose_outputs = decompose(
            self.output_explained_variance, self.output_retained_dimensions
        )

    def unpack_parameters(self):
        # TODO: params from above -> kwargs here?
        setup_params = {  # Pass to the campaign in the library
            "fidelity": self.fidelity,
            "estimator": self.estimator,
            "estimator_kwargs": self.estimator_params.unpack_parameters(),
            "decompose_inputs": self.decompose_inputs,
            "decompose_outputs": self.decompose_outputs,
            "input_explained_variance": self.input_explained_variance,
            "input_retained_dimensions": self.input_retained_dimensions,
            "output_explained_variance": self.output_explained_variance,
            "output_retained_dimensions": self.output_retained_dimensions,
        }
        setup_params = remove_none_values(setup_params)
        train_params = {  # Pass to campaign.fit() in the library
            "train_test_ratio": self.train_test_ratio,
            "model_selection": self.model_selection,
            "model_selection_kwargs": self.model_selection_params.unpack_parameters(),
            "shuffle": self.shuffle,
            "seed": self.seed,
        }
        train_params = remove_none_values(train_params)
        if self.dataset_std is not None:
            train_params["dataset_std_id"] = self.dataset_std.id
        params = {**setup_params, **train_params}
        return params


@typechecked
class ScoreParams(Params):
    """Parameter configuration for scoring a trained emulator.

    Attributes:
        metric (str, optional): Metric used for scoring the performance of an emulator.
            Can be one of:

            - "MSLL": The mean standardised log loss (MSLL), calculated as the mean of the log loss of the emulator minus the mean log loss of a trivial model.
              The log loss is defined as the negative log of the probability of getting the test data value according to a predicted distribution.
              The trivial model is taken to be a Gaussian distribution with mean and standard deviation equal to those of the training data.
              Lower (more negative) scores are better, while positive scores indicate serious problems (a model that is less good than the extremely-trivial naive model).
              The MSLL can be thought of as a measure of how good a model is, as opposed to just taking the average and standard deviation of the training data.
              This is the default metric, and the only metric that accounts for the model uncertainty, which is usually necessary when training a probabilistic model.
            - "MSE": The mean squared error (MSE) is the average of the squared differences between your predicted mean values and those of the test set.
              The MSE quantifies deviations in the model mean predictions only, and is not affected by the model uncertainty estimates.
              A value of zero indicates a model that fits to the data perfectly, but this is not necessarily desirable, as it may indicate overfitting.
            - "RMSE": The root mean squared error (RMSE) is the square root of the MSE and provides a measure of the expected error in the output.
              The RMSE may be considered more interpretable than the MSE, because it shares the same units as the output values.
              However, like the MSE and R2, since model uncertainty is not apart of how this metric is calculated, a desirable RMSE score may belie an underlying poorly-fitting model.
            - "R2": A dimensionless number calculated as one minus the ratio of the MSE to the variance of the test set.
              A value of 1 indicates a perfect model, while a value of 0 indicates a model that is no better than the mean of the test set.
              Negative values are possible but unusual, and indicate a model that is worse than simply taking the mean of the test set.
              As with MSE and RMSE, the model uncertainty is not accounted for in this score; thus, it is possible to have a high R2 score, but a poorly-fitting model.

            The default metric is "MSLL".

        combined_score (bool, optional): Determine whether to combine (average) the emulator score across output dimensions.
            If False, a dataframe of scores will be returned, with the score for each output dimension,
            even if there is only a single emulator output dimension.
            If True, a single number will be returned, which is the average score across all output dimensions.
            The default is ``False``.
    """

    def __init__(
        self,
        metric: str = "MSLL",
        combined_score: bool = False,
    ):
        self.metric = metric
        self.combined_score = combined_score

    def unpack_parameters(self):
        params = {
            "metric": self.metric,
            "combined_score": self.combined_score,
        }
        return params


class BenchmarkParams(Params):
    """Parameter configuration for benchmarking a trained emulator.

    Attributes:
        type (str, optional): Specifies the type of emulator benchmark to be performed.
            Can be one of:

            - ``"quantile"``: The calibration curve is calculated over statistical quantiles.
            - ``"interval"``: The calibration curve is calculated over confidence intervals.

            The default is ``"quantile"``.

        For example, for a well calibrated emulator one would expect to have 10 percent of the unseen datapoints (from the test set) to be outside of the emulator's 90 percent confidence bound.
        If a given confidence interval contains less than expected amount of data the model is underconfident, whereas if it contains more then it is overconfident.
        The calibration curve is necessarily equal to ``0`` and ``1`` at the beginning and end respectively, as the fraction of data within the entire confidence interval must be between 0 and 1.
        Curves are also necessarily monotonically increasing.
        Convex calibration curves (those below the line ``y = x``) indicate that the model is underconfident,
        while concave calibration curves (those above the line ``y = x``) indicate that the model is overconfident.
        It is possible for a curve to be both above and below the line ``y = x``, indicating regions of under- and overconfidence, and possible non-Gaussianity in the data.

        If ``type = "quantile"`` then the calibration curve is calculated over statistical quantiles extending from negative infinity to positive infinity.

        If ``type = "interval"`` then the calibration curve is calculated over confidence intervals,
        starting from the mean of the distribution and extending outwards in both directions simultaneously until the entire confidence interval is covered at negative/positive infinity.

    """

    def __init__(
        self,
        type: str = "quantile",
    ):
        self.type = type

    def unpack_parameters(self):
        params = {"type": self.type}
        return params


@typechecked
class PredictParams(Params):
    """Parameter configuration for making predictions using a trained emulator.

    Attributes:
        observation_noise (bool): Whether or not to include the noise term in the standard deviation of the prediction.
            Setting this to ``False`` can be a good idea if the training data is noisy but the underlying trend of the trained model is smooth.
            In this case, the predictions would correspond to the underlying trend.
            Setting this to ``True`` can be a good idea to see the spread of possible predictions including the noise.
            This is useful to know the true uncertainty in the data-generation process, and to estimate the true span of values that might be observed next.
            The default value is ``True``.

    """

    def __init__(
        self,
        observation_noise: bool = True,
    ):
        self.observation_noise = observation_noise

    def unpack_parameters(self):
        params = {"kwargs": {"observation_noise": self.observation_noise}}
        return params


@typechecked
class SampleParams(Params):
    """Parameter configuration for sampling from a trained emulator.

    Attributes:
        seed (Union[int, None], optional): Specifies the seed used by the random number generator to generate a set of samples.
            Setting this to an integer is useful for the reproducibility of results.
            The default value is ``None``, which means the seed is randomly generated each time.
        observation_noise (bool): Whether or not to include the noise term in the standard deviation of the samples generated.
            Setting this to ``False`` can be a good idea if the training data is noisy but the underlying trend of the trained model is smooth.
            In this case, the samples would look smooth and would model the underlying trend well.
            Setting this to ``True`` can be a good idea to visualise how noisy are the samples from the emulator, which is a consequence of the noise in the observations.
            The default value is ``False``.
        fidelity (Union[pandas.DataFrame, None], optional): Fidelity information to be provided if the model is a multi-fidelity model (``estimator_type="multi_fidelity_gp"`` in ``EstimatorParams``).
            This must be a single column `pandas.DataFrame` with the same sample order as the dataframe of X values used to draw samples.
            The default value is ``None``, which is appropriate for most trained emulators.

    """

    # TODO: Does the fidelity parameter need to be between 0 and 1?

    def __init__(
        self,
        seed: Optional[int] = None,
        observation_noise: bool = False,
        fidelity: Optional[pd.DataFrame] = None,
    ):
        self.seed = seed
        self.observation_noise = observation_noise
        self.fidelity = fidelity

    def unpack_parameters(self):
        params = {
            "kwargs": {
                "seed": self.seed,
                "observation_noise": self.observation_noise,
                "fidelity": self.fidelity,
            }
        }
        params = remove_none_values(params)
        if "fidelity" in params["kwargs"]:
            params["kwargs"]["fidelity"] = _convert_dataframe_to_dict(
                params["kwargs"], "fidelity"
            )
        return params


@deprecated(
    version="2.6.0",
    reason="AcqFuncParams is deprecated and will be removed in a future release. The functionality can be found in the RecommendParams class henceforth.",
)
class AcqFuncParams(Params):
    def __init__():
        pass


@deprecated(
    version="2.6.0",
    reason="OptimiserParams is deprecated and will be removed in a future release. The functionality can be found in the RecommendParams class henceforth.",
)
class OptimiserParams(Params):
    def __init__():
        pass


@typechecked
class RecommendParams(Params):
    """Parameter configuration for recommending new points to sample using the Bayesian-optimisation routine.

    Attributes:
        weights (Union[list[float], None], optional):
            A list of weighting values that are used to create a scalar objective function in the case of a multi-output model.
            The recommend functionality can only work on a single scalar function, so in the case of a multi-output model, the outputs must be combined into a single scalar value.
            The weights create a linear combinations of the outputs, where the weights are the coefficients of the linear combination.
            In the case of a single-output model, the weights are not used and have no impact.
            If the output dimensions are not equally important, the weights can be used to reflect this.
            If the output values have different units, the weights must be chosen to reflect this.
            For example, if two outputs have units of distance and time the weights implicitly have units such that these can be combined (i.e., per metre and per second).
            A list of values is used here, where the first value corresponds to the first output, the second value to the second output, and so on.
            For example, ``[1, 2, 3]`` would create a linear combination of the outputs where the first output is multiplied by 1, the second output by 2, and the third output by 3.
            In this case, the third output is considered to be three times as important as the first output.
            If the values ``[0, 1]`` are used, the first output is ignored and the second output is used as the scalar objective function.
            If the values ``[-1, 0]`` are used, the first output is used as the scalar objective function, and is minimized.
            The default value, ``None``, applies equal weight to each output dimension. This is recommended for functional emulators.
        num_restarts (int, optional): The number of random restarts for optimisation.
            The default value is ``5``.
        raw_samples (int, optional): The number of samples for initialization.
            The default value is ``128``.
        bounds (Union[Tuple, None], optional): The bounds of the input space.
            If this is set to `None` then the bounds are inferred from the range of the training data.
            Otherwise, this must be a dictionary mapping column names to a tuple of lower and upper bounds.
            For example, ``{"x0": (0, 1), "x1": (0, 2)}`` to set boundaries on two input variables ``x0`` and ``x1``.
        seed (Union[int, None], optional): Specifies the seed used by the random number generator to start the optimiser to discover the recommendations.
            Setting this to an integer is good for reproducibility.
            The default value is ``None``, which means the seed is randomly generated each time.

    """

    def __init__(
        self,
        weights: Optional[List[float]] = None,
        # mc_points: Optional[int] = None, # TODO: Cannot be included because tensor
        num_restarts: int = 5,
        raw_samples: int = 128,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        seed: Optional[int] = None,
    ):
        self.weights = weights
        self.mc_points = None
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.bounds = bounds
        self.seed = seed

        if self.bounds is not None:
            self.bounds = pd.DataFrame(self.bounds)
            self.bounds.rename(
                index={0: "lower_bounds", 1: "upper_bounds"}, inplace=True
            )

    def unpack_parameters(self):
        acq_kwargs = {"weights": self.weights, "mc_points": self.mc_points}
        opt_kwargs = {
            "num_restarts": self.num_restarts,
            "raw_samples": self.raw_samples,
            "bounds": self.bounds,
        }
        if "bounds" in opt_kwargs and opt_kwargs["bounds"] is not None:
            opt_kwargs["bounds"] = _convert_dataframe_to_dict(opt_kwargs, "bounds")
        params = {
            "acq_kwargs": acq_kwargs,
            "opt_kwargs": opt_kwargs,
            "kwargs": {
                "return_acq_func_value": True,  # This is fixed to be true here!
                "seed": self.seed,
            },
        }
        params = remove_none_values(params)
        return params


@typechecked
class CalibrateParams(Params):
    """Parameter configuration for inverting a trained emulator to estimate the input parameters that generated a given output.

    Attributes:
        y_std_model (Union[bool, pd.DataFrame], optional): Whether to include model noise covariance in the likelihood.

            - If ``True`` TODO ...
            - If ``False`` TODO ...
            - If a `pandas.DataFrame` is supplied, it must contain the same columns as the set of emulator outputs.

            The default value is ``False``.
        return_summary (bool, optional): Should the result of the inversion be presented as a summary or as the full solution?

            - If ``True`` then return a summary of the inverse solution.
            - If ``False`` return the entire solution in the form of the points sampled.

            The default value is ``True``.
        iterations (int, optional): The number of points to sample in each inversion chain. More points is better.
            The default value is ``10,000``.
        n_chains (int, optional): The number of independent chains to use for the inversion process.
            More is better, so that the solution derived between indepent chains can be compared and convergence can be checked.
            The default value is ``2``.
        force_sequential (bool, optional): TODO
            The default value is False.
        seed (Union[int, None], optional): Specifies the seed used by the random number generator to start the inversion process.
            Setting the seed to an integer is good for reproducibility.
            The default value is ``None``, which means the seed is randomly generated each time.

    """

    # TODO: This docstring needs to be massively improved.

    def __init__(
        self,
        y_std_model: Union[bool, pd.DataFrame] = False,
        # method: Optional[str] = "TinyDA", # TODO: Commented-out as interacts with "method" of use_model_method
        # prior: Optional[str] = "uniform", # TODO: Allow for scipy types
        return_summary: bool = True,
        iterations: int = 10000,
        n_chains: int = 2,
        force_sequential: bool = False,
        seed: Optional[int] = None,
    ):
        self.y_std_model = y_std_model
        # self.method = method
        # self.prior = prior
        self.return_summary = return_summary
        self.iterations = iterations
        self.n_chains = n_chains
        self.force_sequential = force_sequential
        self.seed = seed

    def unpack_parameters(self):
        params = {
            "y_std_model": self.y_std_model,
            # "method": self.method,
            # "prior": self.prior,
            "return_summary": self.return_summary,
            "kwargs": {
                "iterations": self.iterations,
                "n_chains": self.n_chains,
                "force_sequential": self.force_sequential,
                "seed": self.seed,
            },
        }
        params = remove_none_values(params)
        if (
            "y_std_model" in params
        ):  # TODO: y_std_model is not affected by remove_none_values because default is set to False. If we change that (which seems to be the case), we need to update the API in a similar way of get_candidate_points with bounds.
            params["y_std_model"] = _convert_dataframe_to_dict(params, "y_std_model")
        return params


@typechecked
class DesignParams:
    """Parameter configuration to setup an initial experimental or simulations design structure.

    Attributes:
        sampling_method (Sampling, optional): The sampling method to use for the initial design.
            Options are either:

            • ``tl.LatinHypercube``: Populate the initial design space in a clever way such that each dimension, and projection of dimensions, are sampled evenly.
            • ``tl.UniformRandom``: Randomly populate the input space, which is usually a bad idea.

        seed (Union[int, None], optional): The seed used to initialise the random number generators for reproducibility.
            Setting this to an integer is good for creating reproducibile design configurations.
            The default is ``None``, which means the seed is randomly generated each time.

    """

    # TODO: Why no unpack_parameters method?
    def __init__(
        self,
        sampling_method: Sampling = LatinHypercube(),
        seed: Optional[int] = None,
    ):
        self.seed = seed
        self.sampling_method = sampling_method


class MaximizeParams(Params):
    """Parameter configuration for finding the location of the maximum output of your emulator.

    Attributes:
        opt_weights (Union[List[float], None], optional): A list of weighting values that are used to create a scalar objective function in the case of a multi-output model.
            The maximize functionality can only work on a single scalar function, so in the case of a multi-output model, the outputs must be combined into a single scalar value.
            The weights create a linear combinations of the outputs, where the weights are the coefficients of the linear combination.
            In the case of a single-output model, the weights are not used and have no impact.
            If the output dimensions are not equally important, the weights can be used to reflect this.
            If the output values have different units, the weights must be chosen to reflect this.
            For example, if two outputs have units of distance and time the weights implicitly have units such that these can be combined (i.e., per metre and per second).
            A list of values is used here, where the first value corresponds to the first output, the second value to the second output, and so on.
            For example, ``[1, 2, 3]`` would create a linear combination of the outputs where the first output is multiplied by 1, the second output by 2, and the third output by 3.
            In this case, the third output is considered to be three times as important as the first output.
            If the values ``[0, 1]`` are used, the first output is ignored and the second output is used as the scalar objective function.
            If the values ``[-1, 0]`` are used, the first output is used as the scalar objective function, and is minimized.
            The default value, ``None``, applies equal weight to each output dimension. This is recommended for functional emulators.

    """

    def __init__(
        self,
        opt_weights: Optional[List[float]] = None,
    ):
        self.opt_weights = opt_weights

    def unpack_parameters(self):
        params = {"opt_weights": self.opt_weights}
        params = remove_none_values(params)
        return params
