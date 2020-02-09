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
class NBeatsBlock(tf.keras.models.Model):
    def __init__(
        self,
        input_layer: Input,
        fc_width: int,
        fc_layer_num: int,
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

        block = input_layer
        for i in range(fc_layer_num):
            block = Dense(
                fc_width, activation=tf.keras.activations.relu, name=f"{base_name}_fc_layer_{i}"
            )(block)

        backcast_weights, forecast_weights = self.create_output_weights(
            forecast_horizon, lookback_window, trend_order, stack_weights
        )

        backcast_layer = Dense(units=lookback_window, use_bias=False, activation='linear')(block)
        
        backcast_layer = Dense(
            units=backcast_weights.shape[1],
            weights=[backcast_weights],
            activation = 'linear',
            use_bias = False,
            name=f"{base_name}_backcast",
        )(backcast_layer)

        forecast_layer = Dense(units=forecast_horizon, use_bias=False, activation='linear', )(block)
        forecast_layer = Dense(
            units = forecast_weights.shape[1],
            weights=[forecast_weights],
            activation = 'linear',
            use_bias = False,
            name=f"{base_name}_forecast",
        )(forecast_layer)


        # backcast_model = Model(inputs=[backcast_input], outputs=[input_tensor])
        # forecast_model = Model(inputs=[forecast_input], outputs=[forecast_horizon])

        super().__init__(
            inputs=input_layer, outputs=[backcast_layer, forecast_layer], *args, **kwargs
        )
    def build(self, input_shape):
        pass
    def create_output_weights(
        self,
        forecast_horizon: int,
        lookback_window: int,
        trend_order: int,
        stack_weights: StackWeight = StackWeight.none,
    ) -> Tuple[np.array, np.array]:
        stack_weights = StackWeight(stack_weights)

        forward_tvec = np.arange(0, forecast_horizon) / forecast_horizon
        backwards_tvec = np.arange(0, lookback_window)/ lookback_window

        if stack_weights == StackWeight.trend:
            forecast_weights = np.array(
                [forward_tvec ** i for i in range(0, trend_order)]
            ).T
            backcast_weights = np.array(
                [backwards_tvec ** i for i in range(0, trend_order)]
            ).T
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
    "input_layer": test_input,
    "fc_width": 5,
    "fc_layer_num": 4,
    "forecast_horizon": 20,
    "lookback_window": 4,
    "trend_order": 3,
    "stack_weights": StackWeight.trend,
}

test_block = NBeatsBlock(**test_args)
test_2 = test_block(test_input)
# %% [markdown]
# Defining an NBeats stack
# %%
class NBeatsStack(tf.keras.models.Model):
    def __init__(
        self,
        input_layer: Input,
        fc_width: int,
        forecast_horizon: int,
        lookback_window: int,
        trend_order: int,
        block_length: int,
        stack_weights: StackWeight = StackWeight.none,
        *args,
        **kwargs
    ) -> None:

        model = NBeatsBlock(
            input_layer,
            fc_width=fc_width,
            fc_layer_num=0,
            forecast_horizon=forecast_horizon,
            lookback_window=lookback_window,
            trend_order=trend_order,
            stack_weights=stack_weights,
        )

        backcasts = [model.output[0]]
        forecasts = [model.output[1]]

        for i in range(1, block_length):
            input_layer = subtract([model.input, model.output[0]])

            model = model(input_layer)

            backcasts.append(model.output[0])
            forecasts.append(model.output[1])

        super().__init__(inputs=input_layer, outputs=add(forecasts), *args, **kwargs)

# %%
test_stack = NBeatsStack(**test_args, block_length=3)
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
        input_tensor = Input(shape=4)
        self.stack_weight = StackWeight.trend
        self.fc_width=fc_width
        self.forecast_horizon=forecast_horizon
        self.lookback_window=lookback_window
        self.trend_order=trend_order
        self.block_length=block_length
        
        stack = self._create_stack(input_tensor)
        outputs = [stack.output[1]]
        for index in range(stack_length):
            self._swap_stack_weight()
            stack = self._create_stack(stack.outputs[0])(stack)
            outputs.append(stack.output[1])
        output_tensor = add(outputs)
        super().__init__(inputs=[input_tensor], outputs=[output_tensor], *args, **kwargs)

    def _create_stack(self,input_tensor: Input):
        model = NBeatsStack(input_tensor=input_tensor,
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
