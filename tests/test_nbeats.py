import pytest
from nbeats.nbeats import NBeats, NBeatsBlock, NBeatsStack
import tensorflow as tf
from tensorflow import keras


def test_create_block(nbeats_block):

    assert isinstance(nbeats_block,keras.layers.Layer)

def test_build_block(nbeats_block):
    input_layer = keras.Input(shape=(nbeats_block.lookback_window,))
    output = nbeats_block(input_layer)
    assert len(output) == 2
    assert output[0].shape[1] == nbeats_block.lookback_window 
def test_nbeats_block_wrong_input_shape(nbeats_block):
    input_layer  = keras.Input(shape=(nbeats_block.lookback_window + 1,))
    with pytest.raises(Exception):
        nbeats_block(input_layer)