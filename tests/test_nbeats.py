import pytest
from nbeats.nbeats import NBeats, NBeatsBlock, NBeatsStack, StackWeight, test_block
from tensorflow.keras.layers import Layer


@pytest.fixture(scope='module')
def nbeats_test_params():
    test_args = {
    "input_shape": (None, 3),
    "fc_width": 1,
    "forecast_horizon": 2,
    "lookback_window": 3,
    "trend_order": 4,
    "stack_weights": StackWeight.trend,
    }
    return test_args

def test_create_block(nbeats_test_params):
    test_block = NBeatsBlock(**nbeats_test_params)
    assert isinstance(test_block,Layer)

def test_build_block(nbeats_test_params):
    test_block = NBeatsBlock(**nbeats_test_params) 
    test_block.build(input_shape=4)
    assert test_block.input.shape == (None, 4)
