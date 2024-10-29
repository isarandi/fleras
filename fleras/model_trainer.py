import abc
import collections
import time
import warnings

import fleras.callbacks
import fleras.exceptions
import numpy as np
import tensorflow as tf
from fleras.util import attrdict2dict_nested, dict2attrdict_nested
from fleras.util.easydict import EasyDict
from keras import callbacks as callbacks_module
from tensorflow.python.eager import context

try:
    from keras.src.engine import base_layer, data_adapter
    from keras.src.engine.training import _disallow_inside_tf_function, _get_verbosity, \
        _is_tpu_multi_host
    from keras.src.utils import version_utils
except ModuleNotFoundError:
    from keras.engine import base_layer, data_adapter
    from keras.engine.training import _disallow_inside_tf_function, _get_verbosity, \
        _is_tpu_multi_host
    from keras.utils import version_utils


class ModelTrainer(tf.keras.Model, metaclass=abc.ABCMeta):
    def __init__(self, random_seed=0):
        super().__init__()
        self.train_in_inference_mode = False
        self.my_metrics = collections.defaultdict(tf.keras.metrics.Mean)
        self.my_metrics_test = collections.defaultdict(tf.keras.metrics.Mean)
        self.predict_tensor_names = None
        self.timer = Timer()
        self.random_seed = random_seed
        self.distribute_reduction_method = 'sum'
        self.is_testing = False
        self.gradient_accumulation_steps = 1

    def train_step(self, inps):
        # Seed the random generator for this particular step, so that resuming
        # training will also resume at the correct stat of the random number generator
        tf.random.get_global_generator().reset_from_key_counter(
            key=self.random_seed,
            counter=[self.train_counter // self.gradient_accumulation_steps,
                     self.train_counter // self.gradient_accumulation_steps])

        n_replicas = self.distribute_strategy.num_replicas_in_sync

        # Note! using tf.GradientTape(persistent=True) will use more GPU RAM!
        with tf.GradientTape() as tape:
            inps, preds, losses = self._forward_train_and_compute_losses(
                inps, training=not self.train_in_inference_mode)
            # Gradients will be summed up across replicas (not averaged), so we need to divide by
            # the number of replicas to compensate
            repl_scaled_loss = losses['loss'] / n_replicas

        tf.debugging.assert_all_finite(losses['loss'], 'Nonfinite Loss!')

        # grads_and_vars = self.optimizer.compute_gradients(
        #    repl_scaled_loss, var_list=self.trainable_variables, tape=tape)

        # assert all finite
        # for g, v in grads_and_vars:
        #     # print the norm of each gradient
        #     tf.print(v.name, tf.linalg.norm(g))
        #
        # for g, v in reversed(grads_and_vars):
        #     tf.debugging.assert_all_finite(g, f'Nonfinite Gradient for {v.name}!')

        # grad_norms = {v.name+'_norm': tf.linalg.norm(g) for g, v in grads_and_vars}
        # global_norm = tf.linalg.global_norm([g for g, v in grads_and_vars])
        # global_norm_var = tf.linalg.global_norm([v for g, v in grads_and_vars])
        # self.optimizer.apply_gradients(grads_and_vars)

        self.optimizer.minimize(repl_scaled_loss, self.trainable_variables, tape=tape)

        # Metrics
        metrics = self._compute_metrics(inps, preds, training=True)
        metrics.update(losses)
        # metrics['global_norm'] = global_norm
        # metrics['global_norm_var'] = global_norm_var
        # metrics.update(grad_norms)

        if hasattr(self.optimizer, 'learning_rate'):
            metrics['learning_rate'] = self.optimizer.learning_rate

        if hasattr(self.optimizer, 'loss_scale'):
            metrics['loss_scale'] = self.optimizer.loss_scale

        if (hasattr(self.optimizer, 'dynamic_counter') and
                self.optimizer.dynamic_counter is not None):
            metrics['loss_scale_counter'] = self.optimizer.dynamic_counter

        metrics['step_per_sec'] = (
                tf.numpy_function(
                    self.timer.update_and_get_speed, [], tf.float32) /
                (self.gradient_accumulation_steps * n_replicas))

        for name, value in metrics.items():
            self.my_metrics[name].update_state(value)
        return {f'metrics/{k}': v.result() for k, v in self.my_metrics.items()}

    def test_step(self, inps):
        inps, preds = self._forward_test(inps)
        current_metrics = self._compute_metrics(inps, preds, training=False)
        for name, value in current_metrics.items():
            self.my_metrics[name].update_state(value)
        return {f'metrics/{k}': v.result() for k, v in self.my_metrics.items()}

    def predict_step(self, inps):
        inps, preds = self._forward_test(inps)
        tensors = {**inps, **preds}
        if self.predict_tensor_names is None:
            return tensors
        else:
            return {k: tensors[k] for k in self.predict_tensor_names if k in tensors}

    def reset_metrics(self):
        super().reset_metrics()
        if self.is_testing:
            for m in self.my_metrics_test.values():
                m.reset_state()
        elif self.train_counter % self.gradient_accumulation_steps == 0:
            for m in self.my_metrics.values():
                m.reset_state()

    def fit_epochless(
            self, training_data, steps=1, verbose='auto', callbacks=None,
            validation_split=0.0, validation_data=None, shuffle=True, class_weight=None,
            sample_weight=None, initial_step=0, validation_steps=None,
            validation_batch_size=None, validation_freq=1, max_queue_size=10, workers=1,
            use_multiprocessing=False):
        if callbacks is None:
            callbacks = []

        set_optimizer_iteration_counter(self.optimizer, initial_step)

        for cb in callbacks:
            if isinstance(cb, fleras.callbacks.ProgbarLogger):
                cb.set_initial_step(initial_step)

        callbacks = [
            fleras.callbacks.StepAsEpochWrapper(cb)
            if not isinstance(cb, tf.keras.callbacks.History) else cb
            for cb in callbacks]
        callbacks += [
            fleras.callbacks.DummyProgbarLogger(),
            fleras.callbacks.DummyHistory(),
            tf.keras.callbacks.LambdaCallback(
                on_train_begin=lambda logs: self._train_counter.assign(initial_step)),
            fleras.callbacks.TestStateSetter()]

        return super().fit(
            x=training_data, y=None, batch_size=None, epochs=steps, verbose=verbose,
            callbacks=callbacks, validation_split=validation_split, validation_data=validation_data,
            shuffle=shuffle, class_weight=class_weight, sample_weight=sample_weight,
            initial_epoch=initial_step, steps_per_epoch=1, validation_steps=validation_steps,
            validation_batch_size=validation_batch_size, validation_freq=validation_freq,
            max_queue_size=max_queue_size, workers=workers, use_multiprocessing=use_multiprocessing)

    @tf.function
    def _forward_train_and_compute_losses(self, inps, training):
        inps = dict2attrdict_nested(inps)
        inps = self.prepare_inputs(inps, training=True)
        preds = self.forward_train(inps, training)
        losses = self.compute_losses(inps, preds)
        keras_loss = (
            tf.add_n(self.losses) if self.losses else tf.constant(0, dtype=tf.float32))
        losses['loss'] = losses['loss'] + tf.cast(keras_loss, losses['loss'].dtype)
        return attrdict2dict_nested(inps), attrdict2dict_nested(preds), attrdict2dict_nested(losses)

    @tf.function
    def _forward_test(self, inps):
        inps = dict2attrdict_nested(inps)
        inps = self.prepare_inputs(inps, training=False)
        preds = self.forward_test(inps)
        return attrdict2dict_nested(inps), attrdict2dict_nested(preds)

    @tf.function
    def _compute_metrics(self, inps, preds, training):
        inps = dict2attrdict_nested(inps)
        preds = dict2attrdict_nested(preds)
        metrics = self.compute_metrics(inps, preds, training)
        return attrdict2dict_nested(metrics)

    def prepare_inputs(self, inps, training):
        return inps

    @abc.abstractmethod
    def forward_train(self, inps, training):
        pass

    def forward_test(self, inps):
        return self.forward_train(inps, training=False)

    @abc.abstractmethod
    def compute_losses(self, inps, preds):
        pass

    def compute_metrics(self, inps, preds, training):
        return EasyDict()

    def call(self, inputs, training=None, mask=None):
        raise NotImplementedError('A model trainer itself should not be called.')

    def compile(
            self, optimizer="rmsprop", loss=None, metrics=None, loss_weights=None,
            weighted_metrics=None, run_eagerly=None, steps_per_execution=None, jit_compile=None,
            **kwargs):
        super().compile(
            optimizer=optimizer, loss=loss, metrics=metrics, loss_weights=loss_weights,
            weighted_metrics=weighted_metrics, run_eagerly=run_eagerly,
            steps_per_execution=steps_per_execution, jit_compile=jit_compile, **kwargs)
        self.model.optimizer = self.optimizer

    def predict_no_store(
            self, x, batch_size=None, verbose="auto", steps=None, callbacks=None, max_queue_size=10,
            workers=1, use_multiprocessing=False):
        # version_utils.disallow_legacy_graph('Model', 'predict')
        # self._check_call_args('predict')
        # _disallow_inside_tf_function('predict')
        #
        # # TODO(yashkatariya): Cache model on the coordinator for faster prediction.
        # # If running under PSS, then swap it with OneDeviceStrategy so that
        # # execution will run on the coordinator.
        # original_pss_strategy = None
        # if self.distribute_strategy._should_use_with_coordinator:  # pylint:
        # disable=protected-access
        #     original_pss_strategy = self.distribute_strategy
        #     self._distribution_strategy = None
        #
        # # Cluster coordinator is set by `.fit()` and `.evaluate()` which is not
        # # needed in `.predict()` because all the predictions happen on the
        # # coordinator/locally.
        # if self._cluster_coordinator:
        #     self._cluster_coordinator = None
        #
        # outputs = None
        # with self.distribute_strategy.scope():
        #     # Creates a `tf.data.Dataset` and handles batch and epoch iteration.
        #     dataset_types = (data_types.DatasetV1, data_types.DatasetV2)
        #     if (self._in_multi_worker_mode() or _is_tpu_multi_host(
        #             self.distribute_strategy)) and isinstance(x, dataset_types):
        #         try:
        #             options = options_lib.Options()
        #             data_option = options_lib.AutoShardPolicy.DATA
        #             options.experimental_distribute.auto_shard_policy = data_option
        #             x = x.with_options(options)
        #         except ValueError:
        #             warnings.warn('Using Model.predict with '
        #                           'MultiWorkerDistributionStrategy or TPUStrategy and '
        #                           'AutoShardPolicy.FILE might lead to out-of-order result'
        #                           '. Consider setting it to AutoShardPolicy.DATA.')
        #
        #     data_handler = data_adapter.get_data_handler(
        #         x=x,
        #         batch_size=batch_size,
        #         steps_per_epoch=steps,
        #         initial_epoch=0,
        #         epochs=1,
        #         max_queue_size=max_queue_size,
        #         workers=workers,
        #         use_multiprocessing=use_multiprocessing,
        #         model=self,
        #         steps_per_execution=self._steps_per_execution)
        #
        #     # Container that configures and calls `tf.keras.Callback`s.
        #     if not isinstance(callbacks, callbacks_module.CallbackList):
        #         callbacks = callbacks_module.CallbackList(
        #             callbacks,
        #             add_history=True,
        #             add_progbar=verbose != 0,
        #             model=self,
        #             verbose=verbose,
        #             epochs=1,
        #             steps=data_handler.inferred_steps)
        #
        #     self.predict_function = self.make_predict_function()
        #     self._predict_counter.assign(0)
        #     callbacks.on_predict_begin()
        #     # batch_outputs = None
        #     for _, iterator in data_handler.enumerate_epochs():  # Single epoch.
        #         with data_handler.catch_stop_iteration():
        #             for step in data_handler.steps():
        #                 callbacks.on_predict_batch_begin(step)
        #                 tmp_batch_outputs = self.predict_function(iterator)
        #                 if data_handler.should_sync:
        #                     context.async_wait()
        #                 batch_outputs = tmp_batch_outputs  # No error, now safe to assign.
        #
        #                 # !!!!!!! THIS IS THE ONLY DIFFERENCE OVER KERAS DEFAULT PREDICT
        #                 # here we run the callbacks first and only then store the results.
        #                 # This allows the callbacks to change the outputs, e.g. to remove
        #                 # some tensors after processing them in the callback, in order to save
        #                 # memory by not passing them into the concatenation and storage.
        #
        #                 # if outputs is None:
        #                 #     outputs = nest.map_structure(lambda batch_output: [batch_output],
        #                 #                                  batch_outputs)
        #                 # else:
        #                 #     nest.map_structure_up_to(
        #                 #         batch_outputs,
        #                 #         lambda output, batch_output: output.append(batch_output),
        #                 #         outputs, batch_outputs)
        #
        #                 end_step = step + data_handler.step_increment
        #                 callbacks.on_predict_batch_end(end_step, {"outputs": batch_outputs})
        #
        #     # CHANGE: No need to store
        #     #if batch_outputs is None:
        #     #    raise ValueError('Expect x to be a non-empty array or dataset.')
        #     callbacks.on_predict_end()
        #
        # # CHAGNE: No need to store
        # # all_outputs = nest.map_structure_up_to(batch_outputs, concat, outputs)
        #
        # # If originally PSS strategy was used, then replace it back since predict
        # # is running under `OneDeviceStrategy` after the swap and once its done
        # # we need to replace it back to PSS again.
        # if original_pss_strategy is not None:
        #     self._distribution_strategy = original_pss_strategy
        #
        # # CHANGE: No return
        # #return tf_utils.sync_to_numpy_or_python_type(all_outputs)
        #
        # #####

        base_layer.keras_api_gauge.get_cell("predict").set(True)
        version_utils.disallow_legacy_graph("Model", "predict")
        self._check_call_args("predict")
        _disallow_inside_tf_function("predict")

        # TODO(yashkatariya): Cache model on the coordinator for faster
        # prediction.  If running under PSS, then swap it with OneDeviceStrategy
        # so that execution will run on the coordinator.
        original_pss_strategy = None
        if self.distribute_strategy._should_use_with_coordinator:
            original_pss_strategy = self.distribute_strategy
            self._distribution_strategy = None

        # Cluster coordinator is set by `.fit()` and `.evaluate()` which is not
        # needed in `.predict()` because all the predictions happen on the
        # coordinator/locally.
        if self._cluster_coordinator:
            self._cluster_coordinator = None

        verbose = _get_verbosity(verbose, self.distribute_strategy)
        with self.distribute_strategy.scope():
            # Creates a `tf.data.Dataset` and handles batch and epoch iteration.
            dataset_types = (tf.compat.v1.data.Dataset, tf.data.Dataset)
            if (
                    self._in_multi_worker_mode()
                    or _is_tpu_multi_host(self.distribute_strategy)
            ) and isinstance(x, dataset_types):
                try:
                    options = tf.data.Options()
                    data_option = tf.data.experimental.AutoShardPolicy.DATA
                    options.experimental_distribute.auto_shard_policy = (
                        data_option
                    )
                    x = x.with_options(options)
                except ValueError:
                    warnings.warn(
                        "Using Model.predict with MultiWorkerMirroredStrategy "
                        "or TPUStrategy and AutoShardPolicy.FILE might lead to "
                        "out-of-order result. Consider setting it to "
                        "AutoShardPolicy.DATA.",
                        stacklevel=2,
                    )

            data_handler = data_adapter.get_data_handler(
                x=x,
                batch_size=batch_size,
                steps_per_epoch=steps,
                initial_epoch=0,
                epochs=1,
                max_queue_size=max_queue_size,
                workers=workers,
                use_multiprocessing=use_multiprocessing,
                model=self,
                steps_per_execution=self._steps_per_execution,
            )

            # Container that configures and calls `tf.keras.Callback`s.
            if not isinstance(callbacks, callbacks_module.CallbackList):
                callbacks = callbacks_module.CallbackList(
                    callbacks,
                    add_history=True,
                    add_progbar=verbose != 0,
                    model=self,
                    verbose=verbose,
                    epochs=1,
                    steps=data_handler.inferred_steps,
                )

            self.predict_function = self.make_predict_function()
            self._predict_counter.assign(0)
            callbacks.on_predict_begin()
            for _, iterator in data_handler.enumerate_epochs():  # Single epoch.
                with data_handler.catch_stop_iteration():
                    for step in data_handler.steps():
                        callbacks.on_predict_batch_begin(step)
                        tmp_batch_outputs = self.predict_function(iterator)
                        if data_handler.should_sync:
                            context.async_wait()

                        # !!!!!!! THIS IS THE ONLY DIFFERENCE OVER KERAS DEFAULT PREDICT
                        # here we run the callbacks first and only then store the results.
                        # This allows the callbacks to change the outputs, e.g. to remove
                        # some tensors after processing them in the callback, in order to save
                        # memory by not passing them into the concatenation and storage.
                        batch_outputs = (
                            tmp_batch_outputs  # No error, now safe to assign.
                        )
                        end_step = step + data_handler.step_increment
                        callbacks.on_predict_batch_end(
                            end_step, {"outputs": batch_outputs}
                        )

            callbacks.on_predict_end()
        # If originally PSS strategy was used, then replace it back since
        # predict is running under `OneDeviceStrategy` after the swap and once
        # its done we need to replace it back to PSS again.
        if original_pss_strategy is not None:
            self._distribution_strategy = original_pss_strategy

    @property
    def train_counter(self):
        return self._train_counter


class Timer:
    def __init__(self, maxlen=20):
        self.timestamps = collections.deque([time.perf_counter()], maxlen=maxlen + 1)

    def update_and_get_speed(self):
        self.timestamps.append(time.perf_counter())
        timespan = self.timestamps[-1] - self.timestamps[0]
        done_items = len(self.timestamps) - 1
        return np.float32(done_items / timespan)


def set_optimizer_iteration_counter(optimizer, n_completed_steps):
    if hasattr(optimizer, 'iterations'):
        optimizer.iterations.assign(n_completed_steps)

    if hasattr(optimizer, 'inner_optimizer'):
        set_optimizer_iteration_counter(optimizer.inner_optimizer, n_completed_steps)
    #
    #
    # if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
    #     set_optimizer_iteration_counter(optimizer.inner_optimizer, n_completed_steps)
    # # elif isinstance(optimizer, fleras.optimizers.GradientAccumulationOptimizer):
    # #    set_optimizer_iteration_counter(
    # #        optimizer.inner_optimizer, n_completed_steps // optimizer.num_steps)
    # elif isinstance(optimizer, fleras.optimizers.MultiOptimizer):
    #     for opt_spec in optimizer.optimizer_specs:
    #         set_optimizer_iteration_counter(opt_spec['optimizer'], n_completed_steps)
    # else:
    #     optimizer.iterations.assign(n_completed_steps)
