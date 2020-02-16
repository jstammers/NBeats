import pytest
from nbeats.nbeats import StackWeight, NBeatsBlock

def pytest_generate_tests(metafunc):
    if "nbeats_test_params" in metafunc.fixturenames:
        metafunc.parametrize("nbeats_test_params", ["trend", "seasonality"], indirect=True)

@pytest.fixture(scope='module')
def nbeats_test_params(request):
    if request.param == "trend":
        stack_weights = StackWeight.trend
    elif request.param == "seasonality":
        stack_weights = StackWeight.seasonality
    test_args = {
    "input_shape": (None, 3),
    "fc_width": 1,
    "forecast_horizon": 2,
    "lookback_window": 3,
    "stack_weights":stack_weights
    }
    return test_args

@pytest.fixture(scope='module')
def nbeats_block(nbeats_test_params):
    return NBeatsBlock(**nbeats_test_params)