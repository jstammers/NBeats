import pytest
from nbeats.nbeats import StackWeight, NBeatsBlock, NBeatsStack
from nbeats.config import NBeatsConfig


def pytest_generate_tests(metafunc):
    if "nbeats_test_params" in metafunc.fixturenames:
        metafunc.parametrize("nbeats_test_params", ["trend", "seasonality"], indirect=True)


@pytest.fixture(scope='module')
def nbeats_config():
    conf = NBeatsConfig(stack_layers=4,
                        block_layers=4,
                        fc_units=5,
                        fc_layers=4,
                        trend_order=3)
    return conf

@pytest.fixture(scope='module')
def nbeats_test_params(request, nbeats_config):
    if request.param == "trend":
        stack_weights = StackWeight.trend
    elif request.param == "seasonality":
        stack_weights = StackWeight.seasonality

    test_args = {
        "forecast_horizon": 2,
        "lookback_window": 3,
        "stack_weights": stack_weights,
        "config": nbeats_config
    }
    return test_args


@pytest.fixture(scope='module')
def nbeats_block(nbeats_test_params):
    return NBeatsBlock(**nbeats_test_params)


@pytest.fixture(scope='module')
def nbeats_stack(nbeats_test_params):
    return NBeatsStack(**nbeats_test_params)
