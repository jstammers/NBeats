"""nbeats.py 
A representation of the N-Beats model
"""
# %% [markdown]
# ## Load the required modules
# %%
import tensorflow as tf
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import Dense, subtract, add
from enum import Enum
import numpy as np
from typing import List
# %% [markdown]
# The `StackWeight` class is an `Enum` designed to encapsulate the different types of weights that can be applied to a forecast or backcast output
# %%
class StackWeight(Enum):
    none:str = 'none'
    trend:str = 'trend'
    seasonality:str = 'seasonality'
# %% [markdown]
# ## Defining the NBeats model
# %%
class NBeats:
    def __init__(self, forecast_horizon: int, lookback_window: int, block_length: int, stack_length: int, fc_stack_dim: List[int],
    trend_order: int = 3) -> None:
        """Creates an N-Beats model
        
        Parameters
        ----------
        forecast_horizon : int
            The number of samples to forecast
        lookback_window : int
            The number of forecast horizons to use as an input. Typically 2-7
        block_length : int
            The number of basic blocks in a stack layer
        stack_length : int
            The number of stacks in the model
        """
        self.forecast_horizon = forecast_horizon
        self.lookback_window = lookback_window
        self.block_length = block_length
        self.stack_length = stack_length
        self.fc_stack_dim = fc_stack_dim
        self.trend_order = trend_order
        self.model = Model()


    def create_basic_block(self, input_tensor: Input, block_ix: int, stack_ix: int, previous_input = None, stack_weights:str = 'none') -> Model:
        """Creates a basic block layer. This outputs a back-cast of its input as well as a forecast with a length equal to the forecast horizon.
        """

        block = input_tensor
        for fc in self.fc_stack_dim:
            block = Dense(fc, activation=tf.keras.activations.relu)(block)

        backcast_input = Dense(input_tensor.shape[-1],use_bias=False)(block)
        forecast_input = Dense(self.forecast_horizon, use_bias=False, weights=weights[1])(block)

        backcast_model = Model(input = backcast_input, output = input_tensor)
        forecast_model = Model(input = forecast_input, output = self.forecast_horizon)
        return Model(input = input_tensor, output = [backcast_model, forecast_model])

    def create_stack(self, input_tensor:Input, stack_ix: int = 0, stack_weights: str = 'none') -> Model:
        output_weights = self.create_output_weights(stack_weights)
        model = self.create_basic_block(input_tensor, 0,stack_ix)
        backcasts = [model.outputs[0]]
        forecasts = [model.outputs[1]]
        for i in range(1, self.block_length):
            input_tensor = subtract([model.input, model.outputs[0]])
            model = self.create_basic_block(input_tensor, i, stack_ix)
            backcasts.append(model.outputs[0])
            forecasts.append(model.outputs[1])
        return Model(input = input_tensor, output = add(forecasts))

    def create_output_weights(self, stack_weights: str = 'none'):
        stack_weights = StackWeight(stack_weights)
        forward_tvec = np.arange(0,self.forecast_horizon) / self.forecast_horizon
        backwards_tvec = np.arange(0, self.lookback_window)/self.lookback_window
        if stack_weights == StackWeight.trend:
            forecast_weights = np.array([forward_tvec ** i for i in range(0,self.trend_order)])
            backcast_weights = np.array([backwards_tvec ** i for i in range(0, self.trend_order)])
        elif stack_weights == StackWeight.seasonality:
            forecast_weights = [np.ones_like(forward_tvec), ]
        else:
            forecast_weights = None
            backcast_weights = None
        return backcast_weights, forecast_weights
    def show_model(self):
        from keras.utils import plot_model
        plot_model(self.model, to_file='model.png')



# %%
nbeats = NBeats(10,30,3,3,[1,2,3,4])

# %%
bweights, fweights = nbeats.create_output_weights('trend')

# %%


# %%
