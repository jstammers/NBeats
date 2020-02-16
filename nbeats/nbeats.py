"""nbeats.py 
A representation of the N-Beats model
"""
# %% [markdown]
# ## Load the required modules
# %%
import tensorflow as tf
from tensorflow import keras
import nbeats.config as config
from enum import Enum
import numpy as np
from typing import List, Union, Tuple


# %% [markdown]
# The `StackWeight` class is an `Enum` designed to encapsulate the different types of weights that can be applied to a forecast or backcast output
# %%
class StackWeight(Enum):
    none: str = "none"
    trend: str = "trend"
    seasonality: str = "seasonality"


# %% [markdown]
# ## Defining the basic block
# The basic block structure is a fully connected layer
# %%
class NBeatsBlock(keras.layers.Layer):
    def __init__(
        self,
        input_shape: int,
        fc_width: int,
        forecast_horizon: int,
        lookback_window: int,
        stack_weights: StackWeight = StackWeight.none,
        base_name: str = "default",
        *args,
        **kwargs,
    ):
        """Creates a basic block layer. This outputs a back-cast of its input as well as a forecast with a length equal to the forecast horizon.
        """
        super(NBeatsBlock, self).__init__(*args, **kwargs)
        self.stack_weights = stack_weights
        self.fc_width = fc_width
        self.forecast_horizon = forecast_horizon
        self.lookback_window = lookback_window
        self.base_name = base_name
        self.init_layers = [
            keras.layers.Dense(
                self.fc_width,
                activation=tf.keras.activations.relu,
                name=f"{self.base_name}_fc_layer_{i}",
            )
            for i in range(0,config.FC_LAYER_NUM)
        ]
        self.backcast_weights, self.forecast_weights = self.create_output_weights()
        self.backcast_projection = keras.layers.Dense(
            units=self.backcast_weights.shape[0], use_bias=False, activation="linear"
        )
        self.backcast_layer = keras.layers.Dense(
            units=self.lookback_window,
            weights=[self.backcast_weights],
            activation="linear",
            use_bias=False,
            name=f"{self.base_name}_backcast",
        )
        self.forecast_projection = keras.layers.Dense(
            units=self.forecast_weights.shape[0], use_bias=False, activation="linear"
        )

        self.forecast_layer = keras.layers.Dense(
            units=self.forecast_horizon,
            weights=[self.forecast_weights],
            activation="linear",
            use_bias=False,
            name=f"{self.base_name}_forecast",
        )

    def build(self, input_shape):
        super(NBeatsBlock, self).build(input_shape)

    def call(self, input_layer: keras.Input):
        self._check_input_shape(input_layer)
        block = input_layer       
        for layer in self.init_layers:
            block = layer(block)

        backcast_layer = self.backcast_projection(block)

        backcast_layer = self.backcast_layer(backcast_layer)

        forecast_layer = self.forecast_projection(block)

        forecast_layer = self.forecast_layer(forecast_layer)

        return [keras.layers.subtract([backcast_layer, input_layer]), forecast_layer]

    def compute_output_shape(self, input_shape):
        return [self.lookback_window, self.forecast_horizon]

    def create_output_weights(self) -> Tuple[np.array, np.array]:

        forward_tvec = np.arange(0, self.forecast_horizon) / self.forecast_horizon
        backwards_tvec = np.arange(0, self.lookback_window) / self.lookback_window

        if self.stack_weights == StackWeight.trend:
            trend_arr = np.arange(0, config.TREND_ORDER)
            forecast_weights = np.power(forward_tvec[:,np.newaxis],trend_arr).T
            backcast_weights = np.power(backwards_tvec[:,np.newaxis],trend_arr).T
        elif self.stack_weights == StackWeight.seasonality:
            forecast_weights = self.seasonal_coeffs(forward_tvec, self.forecast_horizon)
            backcast_weights = self.seasonal_coeffs(backwards_tvec, self.lookback_window)
        else:
            forecast_weights = None
            backcast_weights = None
        return backcast_weights, forecast_weights

    @staticmethod
    def seasonal_coeffs(time_vec, max_time):
        ones = np.ones_like([time_vec])
        cos = np.cos(
            2 * np.pi * np.arange(0, np.floor(max_time / 2))[:, np.newaxis] * time_vec
        )
        sin = np.sin(
            2 * np.pi * np.arange(0, np.floor(max_time / 2))[:, np.newaxis] * time_vec
        )
        return np.concatenate([ones, cos, sin])

    def _check_input_shape(self, input_layer: keras.Input):
        input_shape = input_layer.shape[1]
        if input_shape != self.lookback_window:
            raise Exception(f"Dimension of input tensor {input_shape} does not match the value expected {self.lookback_window}")
# %% [markdown]
# Defining an NBeats stack
# %%
class NBeatsStack(tf.keras.layers.Layer):
    def __init__(
        self,
        fc_width: int,
        forecast_horizon: int,
        lookback_window: int,
        block_length: int,
        stack_weights: StackWeight = StackWeight.none,
        *args,
        **kwargs,
    ) -> None:
        self.fc_width = fc_width
        self.forecast_horizon = forecast_horizon
        self.lookback_window = lookback_window
        self.stack_weights = stack_weights
        self.block_length = block_length
        super(NBeatsStack, self).__init__(**kwargs)

    def build(self, input_shape):
        super(NBeatsStack, self).build(input_shape)

    def call(self, input_layer):
        model = NBeatsBlock(
            input_shape=input_layer.shape,
            fc_width=self.fc_width,
            forecast_horizon=self.forecast_horizon,
            lookback_window=self.lookback_window,
            trend_order=self.trend_order,
            stack_weights=self.stack_weights,
        )
        backcast, forecast = model(input_layer)
        backcasts = [backcast]
        forecasts = [forecast]
        for i in range(1, self.block_length):
            backcast, forecast = model(backcast)
            backcasts.append(backcast)
            forecasts.append(forecast)
        return [backcasts[-1], keras.layers.add(forecasts)]

    def compute_output_shape(self, input_shape):
        [self.lookback_window, self.forecast_horizon]


# %%


# %% [markdown]
# ## Defining the NBeats model
# %%
class NBeats(keras.Model):
    def __init__(
        self,
        forecast_horizon: int,
        lookback_window: int,
        block_length: int,
        stack_length: int,
        fc_width: int,
        trend_order: int = 3,
        *args,
        **kwargs,
    ) -> None:
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
        input_tensor = keras.Input(shape=(lookback_window,))

        self.stack_weight = StackWeight.trend
        self.fc_width = fc_width
        self.forecast_horizon = forecast_horizon
        self.lookback_window = lookback_window
        self.trend_order = trend_order
        self.block_length = block_length
        backcast, forecast = self._create_stack()(input_tensor)
        outputs = [forecast]
        for index in range(stack_length):
            #  self._swap_stack_weight()
            backcast, forecast = self._create_stack()(backcast)
            outputs.append(forecast)
        output_tensor = keras.layers.add(outputs)
        super().__init__(
            inputs=[input_tensor], outputs=[output_tensor], *args, **kwargs
        )

    def _create_stack(self):
        model = NBeatsStack(
            fc_width=self.fc_width,
            forecast_horizon=self.forecast_horizon,
            lookback_window=self.lookback_window,
            trend_order=self.trend_order,
            stack_weights=self.stack_weight,
            block_length=self.block_length,
        )
        return model

    def _swap_stack_weight(self):
        if self.stack_weight == StackWeight.trend:
            self.stack_weight = StackWeight.seasonality
        elif self.stack_weight == StackWeight.seasonality:
            self.stack_weight = StackWeight.trend
        else:
            raise ValueError("Cannot swap StackWeight")


# %%
if __name__ == "__main__":
    test_input = keras.Input(shape=4)
    test_args = {
        "input_shape": (None, 4),
        "fc_width": 5,
        "forecast_horizon": 20,
        "lookback_window": 4,
        "trend_order": 3,
        "stack_weights": StackWeight.trend,
    }
    test_block = NBeatsBlock(**test_args, dynamic=True)
    test_2 = test_block(tf.ones(shape=(5, 1)))
    test_stack = NBeatsStack(**test_args, block_length=3, dynamic=True)(test_input)
