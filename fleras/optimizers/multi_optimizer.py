# Based on https://github.com/tensorflow/addons/blob/v0.17.1/
# tensorflow_addons/optimizers/discriminative_layer_training.py
#
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
# Copyright 2022 Istvan Sarandi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import tensorflow as tf


class MultiOptimizer(tf.keras.optimizers.Optimizer):
    """Multi Optimizer Wrapper.

    Creates a wrapper around a set of instantiated optimizer layer pairs.
    Generally useful for transfer learning of deep networks.

    Each optimizer will optimize only the weights associated with its paired layer.
    This can be used to implement discriminative layer training by assigning
    different learning rates to each optimizer layer pair.
    `(tf.keras.optimizers.Optimizer, List[tf.keras.layers.Layer])` pairs are also supported.
    Please note that the layers must be instantiated before instantiating the optimizer.

    Args:
        optimizers_and_layers: a list of tuples of an optimizer and a layer or model.
            Each tuple should contain exactly 1 instantiated optimizer and 1 object that
            subclasses `tf.keras.Model`, `tf.keras.Sequential` or `tf.keras.layers.Layer`.
            Nested layers and models will be automatically discovered.
            Alternatively, in place of a single layer, you can pass a list of layers.
        optimizer_specs: specialized list for serialization.
            Should be left as None for almost all cases.
            If you are loading a serialized version of this optimizer,
            please use `tf.keras.models.load_model` after saving a model compiled with this
            optimizer.

    """

    def __init__(self, optimizers_and_layers=None, name='MultiOptimizer', **kwargs):
        super().__init__(name, **kwargs)

        self.optimizer_specs = [
            self.create_optimizer_spec(optimizer, layers_or_model)
            for optimizer, layers_or_model in optimizers_and_layers]


    def apply_gradients(self, grads_and_vars, **kwargs):
        for spec in self.optimizer_specs:
            spec['gv'] = []

        for grad, var in tuple(grads_and_vars):
            for spec in self.optimizer_specs:
                for name in spec['weights']:
                    if var.name == name:
                        spec['gv'].append((grad, var))

        for spec in self.optimizer_specs:
            spec['optimizer'].apply_gradients(spec['gv'], **kwargs)

    def finalize_variable_values(self, variables):
        for spec in self.optimizer_specs:
            spec['optimizer'].finalize_variable_values(
                [v for v in variables if v.name in spec['weights']])

    def get_config(self):
        config = super(MultiOptimizer, self).get_config()
        config.update({'optimizer_specs': self.optimizer_specs})
        return config

    @staticmethod
    def create_optimizer_spec(optimizer, layers_or_model):
        if isinstance(layers_or_model, list):
            weights = [var.name for sublayer in layers_or_model for var in sublayer.weights]
        else:
            weights = [var.name for var in layers_or_model.weights]

        return dict(optimizer=optimizer, weights=weights)

    @property
    def learning_rate(self):
        return self.optimizer_specs[0]['optimizer'].learning_rate
