import abc
import collections
import time

import numpy as np
import tensorflow as tf
from attrdict import AttrDict


class ModelTrainer(tf.keras.Model, metaclass=abc.ABCMeta):
    def __init__(self, global_step, random_seed=None, gradient_accumulation_steps=1):
        super().__init__()
        self.global_step = global_step
        self.train_in_inference_mode = False
        self.my_metrics = {}
        self.predict_tensor_names = None
        self.timer = Timer()
        self.random_seed = random_seed
        self.gradient_accumulation_steps = gradient_accumulation_steps

    def train_step(self, inps):
        # Seed the random generator for this particular step, so that resuming
        # training will also resume at the correct stat of the random number generator
        tf.random.get_global_generator().reset_from_key_counter(
            key=self.random_seed,
            counter=[self.global_step // self.gradient_accumulation_steps,
                     self.global_step // self.gradient_accumulation_steps])

        strategy = tf.distribute.get_strategy()
        n_replicas = strategy.num_replicas_in_sync

        # Note! using tf.GradientTape(persistent=True) will use more GPU RAM!
        with tf.GradientTape() as tape:
            preds = self._forward_train(inps, training=not self.train_in_inference_mode)
            losses = self._compute_losses(inps, preds)
            # Gradients will be summed up across replicas (not averaged), so we need to divide by
            # the number of replicas to compensate
            repl_scaled_loss = losses['loss'] / n_replicas

        tf.debugging.assert_all_finite(losses['loss'], 'Nonfinite Loss!')

        self.optimizer.minimize(repl_scaled_loss, self.trainable_variables, tape=tape)
        self.global_step.assign_add(1)

        # Metrics
        metrics = self._compute_metrics(inps, preds)
        metrics.update(losses)
        if hasattr(self.optimizer, 'learning_rate'):
            metrics['learning_rate'] = self.optimizer.learning_rate

        metrics['step_per_sec'] = (
                tf.numpy_function(
                    self.timer.update_and_get_speed, [], tf.float32) /
                (self.gradient_accumulation_steps * n_replicas))

        return {f'metrics/{k}': v for k, v in metrics.items()}

    def test_step(self, inps):
        preds = self._forward_test(inps)
        current_metrics = self._compute_metrics(inps, preds)
        for metric_name, metric_value in current_metrics.items():
            if metric_name not in self.my_metrics:
                self.my_metrics[metric_name] = tf.keras.metrics.Mean(name=metric_name)
            self.my_metrics[metric_name].update_state(metric_value)
        return {f'metrics/{k}': v.result() for k, v in self.my_metrics.items()}

    def predict_step(self, inps):
        preds = self._forward_test(inps)
        tensors = {**inps, **preds}
        if self.predict_tensor_names is None:
            return tensors
        else:
            return {k: tensors[k] for k in self.predict_tensor_names if k in tensors}

    def reset_metrics(self):
        super().reset_metrics()
        for m in self.my_metrics.values():
            m.reset_state()

    @tf.function
    def _forward_train(self, inps, training):
        inps = AttrDict(inps)
        result = self.forward_train(inps, training)
        result['_keras_loss'] = (
            tf.add_n(self.losses) if self.losses else tf.constant(0, dtype=tf.float32))
        return dict(result)

    @tf.function
    def _forward_test(self, inps):
        inps = AttrDict(inps)
        result = self.forward_test(inps)
        return dict(result)

    @tf.function
    def _compute_losses(self, inps, preds):
        inps = AttrDict(inps)
        preds = AttrDict(preds)
        result = self.compute_losses(inps, preds)
        result['loss'] = result['loss'] + tf.cast(preds['_keras_loss'], result['loss'].dtype)
        return dict(result)

    @tf.function
    def _compute_metrics(self, inps, preds):
        inps = AttrDict(inps)
        preds = AttrDict(preds)
        result = self.compute_metrics(inps, preds)
        return dict(result)

    @abc.abstractmethod
    def forward_train(self, inps, training):
        pass

    def forward_test(self, inps):
        return self.forward_train(inps, training=False)

    @abc.abstractmethod
    def compute_losses(self, inps, preds):
        pass

    @abc.abstractmethod
    def compute_metrics(self, inps, preds):
        pass

    def call(self, inputs, training=None, mask=None):
        raise NotImplementedError('A model trainer itself should not be called.')


class Timer:
    def __init__(self, maxlen=20):
        self.timestamps = collections.deque([time.perf_counter()], maxlen=maxlen + 1)

    def update_and_get_speed(self):
        self.timestamps.append(time.perf_counter())
        timespan = self.timestamps[-1] - self.timestamps[0]
        done_items = len(self.timestamps) - 1
        return np.float32(done_items / timespan)
