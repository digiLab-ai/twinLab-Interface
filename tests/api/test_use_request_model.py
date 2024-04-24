# Project imports
import twinlab.api as api


def test_predict(training_setup, data_csv):
    # NOTE: training_setup is a necessary argument, even though not used, in order to run the fixture
    status, body = api.use_request_model(
        model_id="test_model",
        method="predict",
        data_csv=data_csv,
    )
    assert status == 202
    assert "process_id" in body
    assert body["message"] == "Campaign method predict started"


def test_sample(training_setup, data_csv):
    # NOTE: training_setup is a necessary argument, even though not used, in order to run the fixture
    # The kwargs seems to be incorrect here and that is why tests where failing
    # (We were looking for instances in the kwargs dictionary that doesnt exist here).
    # This is an example of kwargs that comes from a classic emulator.sample.
    # {'num_points': 1, 'acq_func': 'PSD', 'acq_kwargs': {}, 'opt_kwargs': {'num_restarts': 5, 'raw_samples': 128}, 'kwargs': {'return_acq_func_value': True}}
    # Notice that num_samples is part of kwargs and not a direct argument.
    # Also, kwargs used the default SampleParams() class (from where opt_kwargs and acq_kwargs come from)
    status, body = api.use_request_model(
        model_id="test_model",
        method="sample",
        data_csv=data_csv,
        num_samples=3,
        kwargs={"seed": 0},
    )
    assert status == 202
    assert "process_id" in body
    assert body["message"] == "Campaign method sample started"


def test_get_candidate_points(training_setup):
    # NOTE: training_setup is a necessary argument, even though not used, in order to run the fixture
    status, body = api.use_request_model(
        model_id="test_model",
        method="get_candidate_points",
        acq_func="qNIPV",
        num_points=1,
        kwargs={"seed": 0},
    )
    assert status == 202
    assert "process_id" in body
    assert body["message"] == "Campaign method get_candidate_points started"


def test_solve_inverse(training_setup, inverse_csv, inverse_std_csv):
    # NOTE: training_setup is a necessary argument, even though not used, in order to run the fixture
    # Similarly, the kwargs doesnt seem to be correct here and that is why tests where failing.
    # This an example of kwargs that comes from a classic emulator.recommend.
    # {'y_std_model': False, 'return_summary': True, 'kwargs': {'iterations': 10000, 'n_chains': 2, 'force_sequential': False}}
    status, body = api.use_request_model(
        model_id="test_model",
        method="solve_inverse",
        data_csv=inverse_csv,
        data_std_csv=inverse_std_csv,
        kwargs={
            "iterations": 100,
            "force_sequential": "true",
            "seed": 0,
        },
    )
    assert status == 202
    assert "process_id" in body
    assert body["message"] == "Campaign method solve_inverse started"
