import pytest
from nbeats.nbeats import NBeats
from tensorflow.keras import Input

@pytest.fixture(scope='module')
def nbeats():
    return NBeats(forecast_horizon=1, lookback_window=3, block_length=3, stack_length=3, fc_stack_dim=[1,2,3,4])

def test_create_basic_block(nbeats: NBeats):
    input_tensor = Input(shape=(None,1))
    f_layer, b_layer = nbeats.create_basic_block(input_tensor,0,0)
    assert f_layer.shape[-1] == 1
    assert b_layer.shape[-1] == nbeats.forecast_horizon

def test_create_stack(nbeats: NBeats):
    input_tensor = Input(shape=(None,1))
    output = nbeats.create_stack(input_tensor,0)
    


