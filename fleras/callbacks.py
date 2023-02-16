import tensorflow as tf


class WandbCallback(tf.keras.callbacks.Callback):
    def __init__(
            self, global_step_var, logdir, config_dict, project_name, grad_accum_steps=1,
            every_n_steps=30):
        super().__init__()
        self.global_step_var = global_step_var
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
            dir=f'{self.logdir}', id=run_id, resume='allow')

    def on_epoch_end(self, epoch, logs=None):
        import wandb
        step = self.global_step_var.value() / self.grad_accum_steps

        # Only report training batch metrics for every 30th step.
        if step % self.every_n_steps != 0:
            logs = {k: v for k, v in logs.items() if k.startswith('val_')}
        if logs:
            wandb.log(logs, step=int(step), commit=True)


class ProgbarCallback(tf.keras.callbacks.ProgbarLogger):
    def __init__(self, n_completed_steps, n_total_steps):
        super().__init__(count_mode='steps')
        self.n_completed_steps = n_completed_steps
        self.n_total_steps = n_total_steps

    def on_train_begin(self, logs=None):
        super(ProgbarCallback, self).on_train_begin(logs)
        self.seen = self.n_completed_steps
        self.target = self.n_total_steps
        self._maybe_init_progbar()

    def on_epoch_begin(self, epoch, logs=None):
        # We do not use epochs as a concept.
        self.step = epoch
        pass

    def on_train_batch_end(self, batch, logs=None):
        batch = self.step
        if logs is not None:
            # Progbar should not average anything, we want to see raw info
            self.progbar._update_stateful_metrics(list(logs.keys()))
        super(ProgbarCallback, self).on_train_batch_end(batch, logs)

    def on_epoch_end(self, epoch, logs=None):
        # We do not use epochs as a concept.
        pass

    def on_train_end(self, logs=None):
        super(ProgbarCallback, self).on_epoch_end(logs)
        super(ProgbarCallback, self).on_train_end(logs)


class MyProgbar(keras.callbacks.Progbar):
    """Modified keras progbar to support starting (restoring) at an arbitrary step."""

    def __init__(self,
                 target,
                 width=30,
                 verbose=1,
                 interval=0.05,
                 stateful_metrics=None,
                 unit_name='step'):
        super(MyProgbar, self).__init__(
            target, width, verbose, interval, stateful_metrics, unit_name)
        self._initial_step = None

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


# Monkey patch!!
tf.keras.callbacks.Progbar = MyProgbar


class SwitchToInferenceModeCallback(tf.keras.callbacks.Callback):
    def __init__(self, global_step_var, step_to_switch_to_inference_mode, ckpt_manager):
        super().__init__()
        self.global_step_var = global_step_var
        self.step_to_switch_to_inference_mode = step_to_switch_to_inference_mode
        self.ckpt_manager = ckpt_manager

    def on_train_batch_begin(self, batch, logs=None):
        if (self.global_step_var > self.step_to_switch_to_inference_mode
                and not self.model.train_in_inference_mode):
            self.ckpt_manager.save(self.global_step_var, check_interval=False)
            self.model.train_in_inference_mode = True
            self.model.make_train_function(force=True)
