# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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


class AdamW2(tf.keras.optimizers.experimental.AdamW):
    """AdamW optimizer. The difference compared to the basic TensorFlow version is
    that here we accept functions as the learning rate and weight decay arguments
    and they are handled properly. The TF version (in v2.10) calls the learning rate only at
    initialization time and does not allow the weight decay to be a function."""

    def __init__(
            self, learning_rate=0.001, weight_decay=0.004, beta_1=0.9, beta_2=0.999, epsilon=1e-7,
            amsgrad=False, clipnorm=None, clipvalue=None, global_clipnorm=None, use_ema=False,
            ema_momentum=0.99, ema_overwrite_frequency=None, jit_compile=True, name="AdamW",
            **kwargs):
        super(tf.keras.optimizers.experimental.AdamW, self).__init__(
            name=name, clipnorm=clipnorm, clipvalue=clipvalue, global_clipnorm=global_clipnorm,
            use_ema=use_ema, ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency, jit_compile=jit_compile, **kwargs)

        if callable(learning_rate):
            self._learning_rate = learning_rate
            self._current_learning_rate = tf.Variable(
                self._learning_rate(), name='learning_rate', dtype=tf.float32, trainable=False)
        else:
            self._learning_rate = self._build_learning_rate(learning_rate)

        self._weight_decay = weight_decay
        if callable(self._weight_decay):
            self._current_weight_decay = tf.Variable(
                self._weight_decay(), name='weight_decay', dtype=tf.float32, trainable=False)

        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.amsgrad = amsgrad

    def apply_gradients(self, grads_and_vars, **kwargs):
        if callable(self._learning_rate):
            self._current_learning_rate.assign(self._learning_rate())

        if callable(self._weight_decay):
            self._current_weight_decay.assign(self._weight_decay())

        return super().apply_gradients(grads_and_vars, **kwargs)

    @property
    def learning_rate(self):
        if not hasattr(self, "_learning_rate") or self._learning_rate is None:
            raise ValueError(
                "Missing learning rate, please set self.learning_rate at"
                " optimizer creation time.")
        lr = self._learning_rate
        if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule) or callable(lr):
            # If the optimizer takes in LearningRateSchedule, then each call to
            # learning_rate would return `self._current_learning_rate`, which is
            # updated at each call to `apply_gradients`.
            return self._current_learning_rate
        return lr

    @property
    def weight_decay(self):
        if not hasattr(self, "_weight_decay") or self._weight_decay is None:
            raise ValueError(
                "Missing weight decay, please set self.weight_decay at"
                " optimizer creation time.")
        wd = self._weight_decay
        if callable(wd):
            return self._current_weight_decay
        return wd
