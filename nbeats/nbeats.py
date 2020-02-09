"""nbeats.py 
A representation of the N-Beats model
"""
# %% [markdown]
# ## Load the required modules
# %%
import tensorflow as tf
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import Dense, Embedding, subtract, add
from tensorflow.keras import backend as K
from enum import Enum
import numpy as np
from typing import List, Union, Tuple
from tensorflow.keras.utils import plot_model

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
class NBeatsBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        input_shape: int,
        fc_width: int,
        forecast_horizon: int,
        lookback_window: int,
        trend_order: int,
        stack_weights: StackWeight = StackWeight.none,
        base_name: str = 'default',
        *args,
        **kwargs
    ):
        """Creates a basic block layer. This outputs a back-cast of its input as well as a forecast with a length equal to the forecast horizon.
        """
        super(NBeatsBlock, self).__init__(*args, **kwargs)
        self.stack_weights = stack_weights
        self.fc_width=fc_width
        self.forecast_horizon=forecast_horizon
        self.lookback_window=lookback_window
        self.trend_order=trend_order
        self.base_name = base_name
        self.init_layers = [Dense(
                self.fc_width, activation=tf.keras.activations.relu, name=f"{self.base_name}_fc_layer_{i}"
            ) for i in range(4)]

        self.backcast_weights, self.forecast_weights = self.create_output_weights()

        self.backcast_projection = Dense(units=self.trend_order, use_bias=False, activation='linear')
        self.backcast_layer = Dense(
            units=self.lookback_window,
            weights=[self.backcast_weights],
            activation = 'linear',
            use_bias = False,
            name=f"{self.base_name}_backcast",
        )
        self.forecast_projection = Dense(units=self.trend_order, use_bias=False, activation='linear')

        self.forecast_layer = Dense(
            units = self.forecast_horizon,
            weights=[self.forecast_weights],
            activation = 'linear',
            use_bias = False,
            name=f"{self.base_name}_forecast",
        )

    def build(self, input_shape):
        super(NBeatsBlock, self).build(input_shape)

    def call(self, input_layer):
        block = input_layer
        for i in range(4):
            block = self.init_layers[i](block)

        backcast_layer = self.backcast_projection(block)
        
        backcast_layer = self.backcast_layer(backcast_layer)

        forecast_layer = self.forecast_projection(block)

        forecast_layer = self.forecast_layer(forecast_layer)

        return [subtract([backcast_layer,input_layer]), forecast_layer]
    
    def compute_output_shape(self, input_shape):
        return [self.lookback_window, self.forecast_horizon]

    def create_output_weights(
        self) -> Tuple[np.array, np.array]:

        forward_tvec = np.arange(0, self.forecast_horizon) / self.forecast_horizon
        backwards_tvec = np.arange(0, self.lookback_window)/ self.lookback_window

        if self.stack_weights == StackWeight.trend:
            forecast_weights = np.array(
                [forward_tvec ** i for i in range(0, self.trend_order)]
            )
            backcast_weights = np.array(
                [backwards_tvec ** i for i in range(0, self.trend_order)]
            )
        elif self.stack_weights == StackWeight.seasonality:
            forecast_weights = [
                np.ones_like(forward_tvec),
            ]
            backcast_weights = [
                np.ones_like(backwards_tvec),
            ]
        else:
            forecast_weights = None
            backcast_weights = None
        return backcast_weights, forecast_weights


# %%
test_input = Input(shape=4)
test_args = {
    "input_shape": (None,4),
    "fc_width": 5,
    "forecast_horizon": 20,
    "lookback_window": 4,
    "trend_order": 3,
    "stack_weights": StackWeight.trend,
}

test_block = NBeatsBlock(**test_args, dynamic=True)
test_2 = test_block(tf.ones(shape=(5, 1)))
# %% [markdown]
# Defining an NBeats stack
# %%
class NBeatsStack(tf.keras.layers.Layer):
    def __init__(
        self,
        fc_width: int,
        forecast_horizon: int,
        lookback_window: int,
        trend_order: int,
        block_length: int,
        stack_weights: StackWeight = StackWeight.none,
        *args,
        **kwargs
    ) -> None:
        self.fc_width = fc_width
        self.forecast_horizon = forecast_horizon
        self.lookback_window = lookback_window
        self.trend_order = trend_order
        self.stack_weights = stack_weights
        self.block_length = block_length
        super(NBeatsStack, self).__init__(**kwargs)
    
    def build(self, input_shape):
        super(NBeatsStack, self).build(input_shape)

    def call(self, input_layer):
        model = NBeatsBlock(
            input_shape = input_layer.shape,
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
            backcast, forecast  = model(backcast)
            backcasts.append(backcast)
            forecasts.append(forecast)
        return [backcasts[-1],add(forecasts)]


    def compute_output_shape(self, input_shape):
        [self.lookback_window,self.forecast_horizon]

# %%
test_stack = NBeatsStack(**test_args, block_length=3,dynamic=True)(test_input)
# %% [markdown]
# ## Defining the NBeats model
# %%
class NBeats(tf.keras.models.Model):
    def __init__(
        self,
        forecast_horizon: int,
        lookback_window: int,
        block_length: int,
        stack_length: int,
        fc_width: int,
        trend_order: int = 3,
        *args,
        **kwargs
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
        input_tensor = Input(shape=(lookback_window,))

        self.stack_weight = StackWeight.trend
        self.fc_width=fc_width
        self.forecast_horizon=forecast_horizon
        self.lookback_window=lookback_window
        self.trend_order=trend_order
        self.block_length=block_length
        backcast, forecast = self._create_stack()(input_tensor)
        outputs = [forecast]
        for index in range(stack_length):
          #  self._swap_stack_weight()
            backcast, forecast = self._create_stack()(backcast)
            outputs.append(forecast)
        output_tensor = add(outputs)
        super().__init__(inputs=[input_tensor], outputs=[output_tensor], *args, **kwargs)
        
    def _create_stack(self):
        model = NBeatsStack(
        fc_width=self.fc_width,
        forecast_horizon=self.forecast_horizon,
        lookback_window=self.lookback_window,
        trend_order=self.trend_order,
        stack_weights=self.stack_weight,
        block_length=self.block_length)
        return model
    def _swap_stack_weight(self):
        if self.stack_weight == StackWeight.trend:
            self.stack_weight = StackWeight.seasonality
        elif self.stack_weight == StackWeight.seasonality:
            self.stack_weight = StackWeight.trend
        else:
            raise ValueError('Cannot swap StackWeight')
    
# %%
test_nbeats = NBeats(4,12,10,4,100,4)


# %%

# %%


# %%


# %%