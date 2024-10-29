import numpy as np
import tensorflow as tf
import h5py


class Wandb(tf.keras.callbacks.Callback):
    def __init__(self, logdir, config_dict, project_name, grad_accum_steps=1, every_n_steps=30):
        super().__init__()
        self.every_n_steps = every_n_steps
        self.logdir = logdir
        self.config_dict = config_dict
        self.project_name = project_name
        self.grad_accum_steps = grad_accum_steps

    def on_train_begin(self, logs=None):
        import wandb
        id_path = f'{self.logdir}/run_id'
        try:
            with open(id_path) as f:
                run_id = f.read()
        except FileNotFoundError:
            run_id = wandb.util.generate_id()
            with open(id_path, 'w') as f:
                f.write(str(run_id))
                f.flush()

        wandb.init(
            name=self.logdir.split('/')[-1], project=self.project_name, config=self.config_dict,
            dir=f'{self.logdir}', id=run_id, resume='allow',
            settings=wandb.Settings(_service_wait=300))

    def on_train_batch_end(self, batch, logs=None):
        import wandb
        step = batch / self.grad_accum_steps

        # Only report training batch metrics for every 30th step.
        if step % self.every_n_steps != 0:
            logs = {k: v for k, v in logs.items() if k.startswith('val_')}
        if logs:
            wandb.log(logs, step=int(step), commit=True)


class TestStateSetter(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()

    def on_test_begin(self, logs=None):
        self.model.is_testing = True

    def on_test_end(self, logs=None):
        self.model.is_testing = False


class ProgbarLogger(tf.keras.callbacks.ProgbarLogger):
    def __init__(self, grad_accum_steps=1):
        super().__init__(count_mode='steps')
        self.grad_accum_steps = grad_accum_steps
        self.initial_step = 0

    def set_initial_step(self, step):
        self.initial_step = step

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)
        self.seen = self.initial_step
        self._maybe_init_progbar()

    def on_train_batch_end(self, batch, logs=None):
        if logs is not None:
            # Progbar should not average anything, we want to see raw info
            # We already average over the grad_accum_steps in the train_step
            self.progbar._update_stateful_metrics(list(logs.keys()))
        super().on_train_batch_end(batch, logs)

    def on_train_end(self, logs=None):
        super().on_epoch_end(logs)
        super().on_train_end(logs)

    def _maybe_init_progbar(self):
        if self.model:
            # Update the existing stateful metrics as `self.model.metrics` may
            # contain updated metrics after `MetricsContainer` is built in the
            # first train step.
            self.stateful_metrics = self.stateful_metrics.union(
                set(m.name for m in self.model.metrics))

        if self.progbar is None:
            self.progbar = MyProgbar(
                target=self.target, verbose=self.verbose, stateful_metrics=self.stateful_metrics,
                unit_name="step" if self.use_steps else "sample",
                grad_accum_steps=self.grad_accum_steps)

        self.progbar._update_stateful_metrics(self.stateful_metrics)


class MyProgbar(tf.keras.utils.Progbar):
    """Modified keras progbar to support starting (restoring) at an arbitrary step."""

    def __init__(
            self, target, width=30, verbose=1, interval=0.05, stateful_metrics=None,
            unit_name='step', grad_accum_steps=1):
        super(MyProgbar, self).__init__(
            target, width, verbose, interval, stateful_metrics, unit_name)
        self._initial_step = None
        self.grad_accum_steps = grad_accum_steps

    def _estimate_step_duration(self, current, now):
        if self._initial_step is None:
            self._initial_step = current - 1

        if current:
            # Modified this to take into account the _initial_step
            if self._time_after_first_step is not None and current > self._initial_step + 1:
                time_per_unit = (
                        (now - self._time_after_first_step) / (current - (self._initial_step + 1)))
            else:
                time_per_unit = (now - self._start) / (current - self._initial_step)

            if current == self._initial_step + 1:
                self._time_after_first_step = now
            return time_per_unit
        else:
            return 0


class FilterPredictionTensors(tf.keras.callbacks.Callback):
    def __init__(self, names):
        super().__init__()
        self.names = names

    def on_predict_batch_end(self, batch, logs=None):
        outputs = logs['outputs']
        for name in outputs:
            if name not in self.names:
                del outputs[name]


class StorePredictionsAsHDF5(tf.keras.callbacks.Callback):
    def __init__(self, filepath, clear=False):
        super().__init__()
        self.file = h5py.File(filepath, 'w')
        self.is_initialized = False
        self.clear = clear

    def initialize(self, outputs):
        for name, data in outputs.items():
            self.file.create_dataset(name=name, data=data, maxshape=(None, *data.shape[1:]))

        self.is_initialized = True

    def on_predict_batch_end(self, batch, logs=None):
        outputs = logs['outputs']
        if not self.is_initialized:
            self.initialize(outputs)
        else:
            for name, data in outputs.items():
                self.file[name].resize(self.file[name].shape[0] + data.shape[0], axis=0)
                self.file[name][-data.shape[0]:] = data

        if self.clear:
            outputs.clear()

    def on_predict_end(self, logs=None):
        self.file.close()
        self.is_initialized = False


class StorePredictionsAsNPZ(tf.keras.callbacks.Callback):
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
        self.arrays = {}

    def on_predict_batch_end(self, batch, logs=None):
        outputs = logs['outputs']
        for name, data in outputs.items():
            self.arrays[name].append(data)

    def on_predict_end(self, logs=None):
        for name, batches in self.arrays:
            self.arrays[name] = np.concatenate(batches, axis=0)

        np.savez(self.filepath, **self.arrays)


class Checkpoint(tf.keras.callbacks.Callback):
    def __init__(self, ckpt_manager):
        super().__init__()
        self.ckpt_manager = ckpt_manager

    def on_train_batch_end(self, batch, logs=None):
        self.ckpt_manager.save(batch)


class StepAsEpochWrapper(tf.keras.callbacks.Callback):
    def __init__(self, inner_callback):
        super().__init__()
        self.inner_callback = inner_callback

    def set_params(self, params):
        new_params = params.copy()
        new_params['steps'] = params['epochs']
        new_params['epochs'] = 1
        self.inner_callback.set_params(new_params)

    def set_model(self, model):
        self.inner_callback.set_model(model)

    # Epoch handlers are routed to train_batch handlers
    def on_epoch_begin(self, epoch, logs=None):
        return self.inner_callback.on_train_batch_begin(batch=epoch, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        return self.inner_callback.on_train_batch_end(batch=epoch, logs=logs)

    # Train batch handlers are ignored
    def on_train_batch_begin(self, batch, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    # Other handlers are delegated
    def on_predict_begin(self, logs=None):
        return self.inner_callback.on_predict_begin(logs=logs)

    def on_predict_end(self, logs=None):
        return self.inner_callback.on_predict_end(logs=logs)

    def on_test_begin(self, logs=None):
        return self.inner_callback.on_test_begin(logs=logs)

    def on_test_end(self, logs=None):
        return self.inner_callback.on_test_end(logs=logs)

    def on_train_begin(self, logs=None):
        return self.inner_callback.on_train_begin(logs=logs)

    def on_train_end(self, logs=None):
        return self.inner_callback.on_train_end(logs=logs)

    def on_predict_batch_begin(self, batch, logs=None):
        return self.inner_callback.on_predict_batch_begin(batch=batch, logs=logs)

    def on_predict_batch_end(self, batch, logs=None):
        return self.inner_callback.on_predict_batch_end(batch=batch, logs=logs)

    def on_test_batch_begin(self, batch, logs=None):
        return self.inner_callback.on_test_batch_begin(batch=batch, logs=logs)

    def on_test_batch_end(self, batch, logs=None):
        return self.inner_callback.on_test_batch_end(batch=batch, logs=logs)


class SwitchToInferenceModeCallback(tf.keras.callbacks.Callback):
    def __init__(self, step_to_switch_to_inference_mode, ckpt_manager):
        super().__init__()
        self.step_to_switch_to_inference_mode = step_to_switch_to_inference_mode
        self.ckpt_manager = ckpt_manager

    def on_train_batch_begin(self, batch, logs=None):
        if (batch > self.step_to_switch_to_inference_mode
                and not self.model.train_in_inference_mode):
            self.ckpt_manager.save(batch, check_interval=False)
            self.ckpt_manager.checkpoint.save('ckpt_before_switch_to_inference_mode')
            self.model.train_in_inference_mode = True
            self.model.make_train_function(force=True)


class DummyHistory(tf.keras.callbacks.History):
    # Keras adds a history callback in model.fit unless there already is one in the
    # supplied callbacks. We don't want to have this, so we insert a dummy history callback
    # that prevents Keras from adding it.
    def __init__(self):
        super().__init__()

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_train_batch_begin(self, batch, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    def on_predict_begin(self, logs=None):
        pass

    def on_predict_end(self, logs=None):
        pass

    def on_test_begin(self, logs=None):
        pass

    def on_test_end(self, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_predict_batch_begin(self, batch, logs=None):
        pass

    def on_predict_batch_end(self, batch, logs=None):
        pass

    def on_test_batch_begin(self, batch, logs=None):
        pass

    def on_test_batch_end(self, batch, logs=None):
        pass


class DummyProgbarLogger(tf.keras.callbacks.ProgbarLogger):
    def __init__(self):
        super().__init__()

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_train_batch_begin(self, batch, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    def on_predict_begin(self, logs=None):
        pass

    def on_predict_end(self, logs=None):
        pass

    def on_test_begin(self, logs=None):
        pass

    def on_test_end(self, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_predict_batch_begin(self, batch, logs=None):
        pass

    def on_predict_batch_end(self, batch, logs=None):
        pass

    def on_test_batch_begin(self, batch, logs=None):
        pass

    def on_test_batch_end(self, batch, logs=None):
        pass
