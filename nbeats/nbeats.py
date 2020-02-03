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
        input_layer: Input,
        fc_width: int,
        fc_layer_num: int,
        forecast_horizon: int,
        lookback_window: int,
        trend_order: int,
        stack_weights: StackWeight = StackWeight.none,
        *args,
        **kwargs
    ):
        """Creates a basic block layer. This outputs a back-cast of its input as well as a forecast with a length equal to the forecast horizon.
        """

        block = input_layer
        for _ in range(fc_layer_num):
            block = Dense(
                fc_width, activation=tf.keras.activations.relu, name="fc_layer"
            )(block)

        backcast_weights, forecast_weights = self.create_output_weights(
            forecast_horizon, lookback_window, trend_order, stack_weights
        )

        backcast_layer = Dense(units=lookback_window, use_bias=False, activation='linear')(block)
        
        backcast_layer = Dense(
            units=backcast_weights.shape[0],
            weights=[backcast_weights.T],
            activation = 'linear',
            use_bias = False,
            name="backcast",
        )(backcast_layer)

        forecast_layer = Dense(units=forecast_horizon, use_bias=False, activation='linear')(block)
        forecast_layer = Dense(
            units = forecast_weights.shape[0],
            weights=[forecast_weights.T],
            activation = 'linear',
            use_bias = False,
            name="forecast",
        )(forecast_layer)


        # backcast_model = Model(inputs=[backcast_input], outputs=[input_tensor])
        # forecast_model = Model(inputs=[forecast_input], outputs=[forecast_horizon])

        super().__init__(
            inputs=[input_layer], outputs=[backcast_layer, forecast_layer], *args, **kwargs
        )
    def build(self, input_shape):
        
    def create_output_weights(
        self,
        forecast_horizon: int,
        lookback_window: int,
        trend_order: int,
        stack_weights: StackWeight = StackWeight.none,
    ) -> Tuple[np.array, np.array]:
        stack_weights = StackWeight(stack_weights)

        forward_tvec = np.arange(0, forecast_horizon).T / forecast_horizon
        backwards_tvec = np.arange(0, lookback_window).T / lookback_window

        if stack_weights == StackWeight.trend:
            forecast_weights = np.array(
                [forward_tvec ** i for i in range(0, trend_order)]
            )
            backcast_weights = np.array(
                [backwards_tvec ** i for i in range(0, trend_order)]
            )
        elif stack_weights == StackWeight.seasonality:
            forecast_weights = [
                np.ones_like(forward_tvec),
            ]
        else:
            forecast_weights = None
            backcast_weights = None
        return backcast_weights, forecast_weights


# %%
test_input = Input(shape=4)
test_args = {
    "input_tensor": test_input,
    "fc_width": 5,
    "fc_layer_num": 4,
    "forecast_horizon": 20,
    "lookback_window": 4,
    "trend_order": 3,
    "stack_weights": StackWeight.trend,
}

test_block = NBeatsBlock(**test_args)
# %% [markdown]
# Defining an NBeats stack
# %%
class NBeatsStack(tf.keras.layers.Layer):
    def __init__(
        self,
        input_tensor: Input,
        layer_units: List[int],
        forecast_horizon: int,
        lookback_window: int,
        trend_order: int,
        stack_weights: StackWeight = StackWeight.none,
        *args,
        **kwargs
    ) -> None:

        model = NBeatsBlock(
            input_tensor,
            layer_units=layer_units,
            forecast_horizon=forecast_horizon,
            lookback_window=lookback_window,
            trend_order=trend_order,
            stack_weights=stack_weights,
        )

        backcasts = [model.outputs[0]]
        forecasts = [model.outputs[1]]

        for i in range(1, self.block_length):
            input_tensor = subtract([model.input, model.outputs[0]])

            model = NBeatsBlock(
                input_tensor,
                layer_units=layer_units,
                forecast_horizon=forecast_horizon,
                lookback_window=lookback_window,
                trend_order=trend_order,
                stack_weights=stack_weights,
            )

            backcasts.append(model.outputs[0])
            forecasts.append(model.outputs[1])

        super().__init__(input=input_tensor, output=add(forecasts), *args, **kwargs)


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
        fc_stack_dim: List[int],
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
        self.forecast_horizon = forecast_horizon
        self.lookback_window = lookback_window
        self.block_length = block_length
        self.stack_length = stack_length
        self.fc_stack_dim = fc_stack_dim
        self.trend_order = trend_order
        super().__init__(*args, **kwargs)

    def show_model(self):
        from keras.utils import plot_model

        plot_model(self.model, to_file="model.png")


# %%
nbeats = NBeats(10, 30, 3, 3, [1, 2, 3, 4])

# %%
bweights, fweights = nbeats.create_output_weights("trend")

# %%


# %%

