# Batch Renormalization Layer that uses fused operations unlike the default TF version
# Copyright
# 2023 Istvan Sarandi
# 2019 The TensorFlow Authors. All Rights Reserved.
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
"""The V2 implementation of Normalization layers."""

import warnings

import tensorflow as tf

try:
    from keras.src.utils import control_flow_util, tf_utils
except ModuleNotFoundError:
    from keras.utils import control_flow_util, tf_utils

backend = tf.keras.backend



class BatchNormalization(tf.keras.layers.Layer):
    r"""Layer that normalizes its inputs.

    Batch normalization applies a transformation that maintains the mean output
    close to 0 and the output standard deviation close to 1.

    Importantly, batch normalization works differently during training and
    during inference.

    **During training** (i.e. when using `fit()` or when calling the layer/model
    with the argument `training=True`), the layer normalizes its output using
    the mean and standard deviation of the current batch of inputs. That is to
    say, for each channel being normalized, the layer returns
    `gamma * (batch - mean(batch)) / sqrt(var(batch) + epsilon) + beta`, where:

    - `epsilon` is small constant (configurable as part of the constructor
    arguments)
    - `gamma` is a learned scaling factor (initialized as 1), which
    can be disabled by passing `scale=False` to the constructor.
    - `beta` is a learned offset factor (initialized as 0), which
    can be disabled by passing `center=False` to the constructor.

    **During inference** (i.e. when using `evaluate()` or `predict()`) or when
    calling the layer/model with the argument `training=False` (which is the
    default), the layer normalizes its output using a moving average of the
    mean and standard deviation of the batches it has seen during training. That
    is to say, it returns
    `gamma * (batch - self.moving_mean) / sqrt(self.moving_var+epsilon) + beta`.

    `self.moving_mean` and `self.moving_var` are non-trainable variables that
    are updated each time the layer in called in training mode, as such:

    - `moving_mean = moving_mean * momentum + mean(batch) * (1 - momentum)`
    - `moving_var = moving_var * momentum + var(batch) * (1 - momentum)`

    As such, the layer will only normalize its inputs during inference
    *after having been trained on data that has similar statistics as the
    inference data*.

    Args:
      axis: Integer or a list of integers, the axis that should be normalized
        (typically the features axis). For instance, after a `Conv2D` layer with
        `data_format="channels_first"`, set `axis=1` in `BatchNormalization`.
      momentum: Momentum for the moving average.
      epsilon: Small float added to variance to avoid dividing by zero.
      center: If True, add offset of `beta` to normalized tensor. If False,
        `beta` is ignored.
      scale: If True, multiply by `gamma`. If False, `gamma` is not used. When
        the next layer is linear (also e.g. `nn.relu`), this can be disabled
        since the scaling will be done by the next layer.
      beta_initializer: Initializer for the beta weight.
      gamma_initializer: Initializer for the gamma weight.
      moving_mean_initializer: Initializer for the moving mean.
      moving_variance_initializer: Initializer for the moving variance.
      beta_regularizer: Optional regularizer for the beta weight.
      gamma_regularizer: Optional regularizer for the gamma weight.
      beta_constraint: Optional constraint for the beta weight.
      gamma_constraint: Optional constraint for the gamma weight.
      renorm_clipping: A dictionary that may map keys 'rmax', 'rmin', 'dmax' to
        scalar `Tensors` used to clip the renorm correction. The correction `(r,
        d)` is used as `corrected_value = normalized_value * r + d`, with `r`
        clipped to [rmin, rmax], and `d` to [-dmax, dmax]. Missing rmax, rmin,
        dmax are set to inf, 0, inf, respectively.
      renorm_momentum: Momentum used to update the moving means and standard
        deviations with renorm. Unlike `momentum`, this affects training and
        should be neither too small (which would add noise) nor too large (which
        would give stale estimates). Note that `momentum` is still applied to
        get the means and variances for inference.
      trainable: Boolean, if `True` the variables will be marked as trainable.

    Call arguments:
      inputs: Input tensor (of any rank).
      training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode.
        - `training=True`: The layer will normalize its inputs using the mean
          and variance of the current batch of inputs.
        - `training=False`: The layer will normalize its inputs using the mean
          and variance of its moving statistics, learned during training.
      mask: Binary tensor of shape broadcastable to `inputs` tensor, indicating
        the positions for which the mean and variance should be computed.

    Input shape: Arbitrary. Use the keyword argument `input_shape` (tuple of
      integers, does not include the samples axis) when using this layer as the
      first layer in a model.

    Output shape: Same shape as input.

    Reference:
      - [Ioffe and Szegedy, 2015](https://arxiv.org/abs/1502.03167).
    """

    # By default, the base class uses V2 behavior. The BatchNormalization V1
    # subclass sets this to False to use the V1 behavior.

    def __init__(
            self,
            axis=-1,
            momentum=0.99,
            epsilon=1e-3,
            center=True,
            scale=True,
            beta_initializer="zeros",
            gamma_initializer="ones",
            moving_mean_initializer="zeros",
            moving_variance_initializer="ones",
            beta_regularizer=None,
            gamma_regularizer=None,
            beta_constraint=None,
            gamma_constraint=None,
            renorm_clipping=None,
            trainable=True,
            name=None,
            **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        if isinstance(axis, (list, tuple)):
            self.axis = axis[:]
        elif isinstance(axis, int):
            self.axis = axis
        else:
            raise TypeError(
                "Expected an int or a list/tuple of ints for the "
                "argument 'axis', but received: %r" % axis
            )

        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.moving_mean_initializer = tf.keras.initializers.get(moving_mean_initializer)
        self.moving_variance_initializer = tf.keras.initializers.get(moving_variance_initializer)
        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = tf.keras.constraints.get(beta_constraint)
        self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)
        self._raise_if_fused_cannot_be_used()
        self.supports_masking = True

        self._bessels_correction_test_only = True
        self.trainable = trainable

        renorm_clipping = renorm_clipping or {}
        keys = ["rmax", "rmin", "dmax"]
        if set(renorm_clipping) - set(keys):
            raise ValueError(
                "Received invalid keys for `renorm_clipping` argument: "
                f"{renorm_clipping}. Supported values: {keys}."
            )
        self.renorm_clipping = renorm_clipping

    def _raise_if_fused_cannot_be_used(self):
        """Raises a ValueError if fused implementation cannot be used.

        In addition to the checks done in this function, the input tensors rank
        must be 4 or 5. The input rank check can only be done once the input
        shape is known.
        """
        # Note the ValueErrors in this function are caught and not reraised in
        # _fused_can_be_used(). No other exception besides ValueError should be
        # raised here.

        # Currently fused batch norm doesn't support renorm. It also only
        # supports a channel dimension on axis 1 or 3 (rank=4) / 1 or 4 (rank5),
        # when no virtual batch size or adjustment is used.
        axis = [self.axis] if isinstance(self.axis, int) else self.axis
        # Axis -3 is equivalent to 1, and axis -1 is equivalent to 3, when the
        # input rank is 4. Similarly, the valid axis is -4, -1, 1, 4 when the
        # rank is 5. The combination of ranks and axes will be checked later.
        if len(axis) > 1 or axis[0] not in (-4, -3, -1, 1, 3, 4):
            raise ValueError(
                "Fused is only supported when axis is 1 "
                "or 3 for input rank = 4 or 1 or 4 for input rank = 5. "
                "Got axis %s" % (axis,)
            )
        # TODO(reedwm): Support fp64 in FusedBatchNorm then remove this check.
        if self._compute_dtype not in ("float16", "bfloat16", "float32", None):
            raise ValueError(
                "Fused is only supported when the compute "
                "dtype is float16, bfloat16, or float32. Got dtype: %s"
                % (self._compute_dtype,)
            )

    @property
    def trainable(self):
        return self._trainable

    @trainable.setter
    def trainable(self, value):
        self._trainable = value

    @property
    def _param_dtype(self):
        # Raise parameters of fp16 batch norm to fp32
        if self.dtype == tf.float16 or self.dtype == tf.bfloat16:
            return tf.float32
        else:
            return self.dtype or tf.float32

    def build(self, input_shape):
        self.axis = tf_utils.validate_axis(self.axis, input_shape)
        input_shape = tf.TensorShape(input_shape)
        rank = input_shape.rank

        if rank not in (4, 5):
            raise ValueError(
                "Batch normalization layers with `fused=True` only "
                "support 4D or 5D input tensors. "
                f"Received tensor with shape: {tuple(input_shape)}"
            )

        if self.axis == [1] and rank == 4:
            self._data_format = "NCHW"
        elif self.axis == [1] and rank == 5:
            self._data_format = "NCDHW"
        elif self.axis == [3] and rank == 4:
            self._data_format = "NHWC"
        elif self.axis == [4] and rank == 5:
            self._data_format = "NDHWC"
        elif rank == 5:
            raise NotImplementedError
        else:
            if rank == 4:
                raise ValueError(
                    "Unsupported axis. The use of `fused=True` is only "
                    "possible with `axis=1` or `axis=3` for 4D input "
                    f"tensors. Received: axis={tuple(self.axis)}"
                )
            else:
                raise ValueError(
                    "Unsupported axis. The use of `fused=True` is only "
                    "possible with `axis=1` or `axis=4` for 5D input "
                    f"tensors. Received: axis={tuple(self.axis)}"
                )

        axis_to_dim = {x: input_shape.dims[x].value for x in self.axis}
        for x in axis_to_dim:
            if axis_to_dim[x] is None:
                raise ValueError(
                    "Input has undefined `axis` dimension. Received input "
                    f"with shape {tuple(input_shape)} "
                    f"and axis={tuple(self.axis)}"
                )
        self.input_spec = tf.keras.layers.InputSpec(ndim=rank, axes=axis_to_dim)

        if len(axis_to_dim) == 1:
            # Single axis batch norm (most common/default use-case)
            param_shape = (list(axis_to_dim.values())[0],)
        else:
            # Parameter shape is the original shape but with 1 in all non-axis
            # dims
            param_shape = [
                axis_to_dim[i] if i in axis_to_dim else 1 for i in range(rank)
            ]
        self._param_shape = param_shape
        if self.scale:
            self.gamma = self.add_weight(
                name="gamma",
                shape=param_shape,
                dtype=self._param_dtype,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                trainable=True,
                experimental_autocast=False,
            )
        else:
            self.gamma = None

        if self.center:
            self.beta = self.add_weight(
                name="beta",
                shape=param_shape,
                dtype=self._param_dtype,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                trainable=True,
                experimental_autocast=False,
            )
        else:
            self.beta = None

        try:
            # Disable variable partitioning when creating the moving mean and
            # variance
            if hasattr(self, "_scope") and self._scope:
                partitioner = self._scope.partitioner
                self._scope.set_partitioner(None)
            else:
                partitioner = None
            self.moving_mean = self.add_weight(
                name="moving_mean",
                shape=param_shape,
                dtype=self._param_dtype,
                initializer=self.moving_mean_initializer,
                synchronization=tf.VariableSynchronization.ON_READ,
                trainable=False,
                aggregation=tf.VariableAggregation.MEAN,
                experimental_autocast=False,
            )

            self.moving_variance = self.add_weight(
                name="moving_variance",
                shape=param_shape,
                dtype=self._param_dtype,
                initializer=self.moving_variance_initializer,
                synchronization=tf.VariableSynchronization.ON_READ,
                trainable=False,
                aggregation=tf.VariableAggregation.MEAN,
                experimental_autocast=False,
            )

        finally:
            if partitioner:
                self._scope.set_partitioner(partitioner)
        self.built = True

    def call(self, inputs, training=None, mask=None):
        inputs = tf.cast(inputs, self.compute_dtype)
        training = self._get_training_value(training)
        # Determine a boolean value for `training`: could be True, False, or None.
        training_value = control_flow_util.constant_value(training)
        return self._fused_batch_norm(inputs, mask=mask, training=training_value)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        initializers = tf.keras.initializers
        regularizers = tf.keras.regularizers
        constraints = tf.keras.constraints
        config = {
            "axis": self.axis, "momentum": self.momentum, "epsilon": self.epsilon,
            "center": self.center, "scale": self.scale,
            "beta_initializer": initializers.serialize(self.beta_initializer),
            "gamma_initializer": initializers.serialize(self.gamma_initializer),
            "moving_mean_initializer": initializers.serialize(self.moving_mean_initializer),
            "moving_variance_initializer": initializers.serialize(self.moving_variance_initializer),
            "beta_regularizer": regularizers.serialize(self.beta_regularizer),
            "gamma_regularizer": regularizers.serialize(self.gamma_regularizer),
            "beta_constraint": constraints.serialize(self.beta_constraint),
            "gamma_constraint": constraints.serialize(self.gamma_constraint),
            "renorm_clipping": self.renorm_clipping}

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    ######################## Start of private methods ##########################
    def _support_zero_size_input(self):
        if not tf.distribute.has_strategy():
            return False
        strategy = tf.distribute.get_strategy()
        # TODO(b/195085185): remove experimental_enable_get_next_as_optional
        # after migrating all users.
        return getattr(
            strategy.extended,
            "enable_partial_batch_handling",
            getattr(
                strategy.extended,
                "experimental_enable_get_next_as_optional",
                False,
            ),
        )

    def _assign_new_value(self, variable, value):
        with backend.name_scope("AssignNewValue") as scope:
            if tf.compat.v1.executing_eagerly_outside_functions():
                return variable.assign(value, name=scope)
            else:
                with tf.compat.v1.colocate_with(variable):
                    return tf.compat.v1.assign(variable, value, name=scope)

    def _assign_moving_average(self, variable, value, momentum, inputs_size):
        def calculate_update_delta():
            decay = tf.convert_to_tensor(1.0 - momentum, name="decay")
            if decay.dtype != variable.dtype.base_dtype:
                decay = tf.cast(decay, variable.dtype.base_dtype)
            update_delta = (variable - tf.cast(value, variable.dtype)) * decay
            if inputs_size is not None:
                update_delta = tf.where(
                    inputs_size > 0, update_delta, backend.zeros_like(update_delta))
            return update_delta

        with backend.name_scope("AssignMovingAvg") as scope:
            if tf.compat.v1.executing_eagerly_outside_functions():
                return variable.assign_sub(calculate_update_delta(), name=scope)
            else:
                with tf.compat.v1.colocate_with(variable):
                    return tf.compat.v1.assign_sub(
                        variable, calculate_update_delta(), name=scope)

    def _fused_batch_norm(self, inputs, mask, training):
        """Returns the output of fused batch norm."""
        if mask is not None:
            warnings.warn(
                "Masking is not supported with `fused=True`. "
                "You should either turn off fusing "
                "(`fused=False`) or you should not pass a `mask` "
                "argument when calling the layer. "
                "For the moment `mask` will be ignored for the "
                "normalization.")
        if self.center:
            beta = self.beta
        else:
            beta = backend.constant(0.0, dtype=self._param_dtype, shape=self._param_shape)
        if self.scale:
            gamma = self.gamma
        else:
            gamma = backend.constant(1.0, dtype=self._param_dtype, shape=self._param_shape)

        # TODO(b/129279393): Support zero batch input in non
        # DistributionStrategy code as well.
        if self._support_zero_size_input():
            # Keras assumes that batch dimension is the first dimension for
            # Batch Normalization.
            input_batch_size = tf.shape(inputs)[0]
        else:
            input_batch_size = None

        def _maybe_add_or_remove_bessels_correction(variance, remove=True):
            r"""Add or remove Bessel's correction."""
            # Removes Bessel's correction if remove == True, adds it otherwise.
            # This is to be consistent with non-fused batch norm. Note that the
            # variance computed by fused batch norm is with Bessel's correction.
            # This is only used in legacy V1 batch norm tests.
            if self._bessels_correction_test_only:
                return variance
            sample_size = tf.cast(tf.size(inputs) / tf.size(variance), variance.dtype)
            if remove:
                factor = (sample_size - tf.cast(1.0, variance.dtype)) / sample_size
            else:
                factor = sample_size / (sample_size - tf.cast(1.0, variance.dtype))
            return variance * factor

        def _training_renorm():
            output, mean, variance = tf.compat.v1.nn.fused_batch_norm(
                inputs, gamma, tf.zeros_like(beta),  # ZERO BETA! because we will renorm-adjust it
                epsilon=self.epsilon, is_training=True, data_format=self._data_format)
            variance = tf.stop_gradient(variance)
            mean = tf.stop_gradient(mean)
            variance = _maybe_add_or_remove_bessels_correction(variance, remove=True)
            moving_var_plus_eps = self.moving_variance + self.epsilon
            r = tf.sqrt((variance + self.epsilon) / moving_var_plus_eps)
            d = (mean - self.moving_mean) * tf.math.rsqrt(moving_var_plus_eps)
            r = tf.clip_by_value(r, self.renorm_clipping['rmin'], self.renorm_clipping['rmax'])
            d = tf.clip_by_value(d, -self.renorm_clipping['dmax'], self.renorm_clipping['dmax'])
            output = output * tf.cast(r, output.dtype) + tf.cast(d * gamma + beta, output.dtype)
            return output, mean, variance

        output, mean, variance = control_flow_util.smart_cond(
            training,
            _training_renorm,
            lambda: tf.compat.v1.nn.fused_batch_norm(
                inputs, gamma, beta, mean=self.moving_mean, variance=self.moving_variance,
                epsilon=self.epsilon, is_training=False, data_format=self._data_format))

        training_value = control_flow_util.constant_value(training)
        if training_value or training_value is None:
            if training_value is None:
                momentum = control_flow_util.smart_cond(
                    training, lambda: self.momentum, lambda: 1.0)
            else:
                momentum = tf.convert_to_tensor(self.momentum)

            def mean_update():
                return self._assign_moving_average(
                    self.moving_mean, mean, momentum, input_batch_size)

            def variance_update():
                return self._assign_moving_average(
                    self.moving_variance, variance, momentum, input_batch_size)

            self.add_update(mean_update)
            self.add_update(variance_update)

        return output

    def _get_training_value(self, training=None):
        if training is None:
            training = backend.learning_phase()
        if isinstance(training, int):
            training = bool(training)
        if not self.trainable:
            # When the layer is not trainable, it overrides the value passed
            # from model.
            training = False
        return training
