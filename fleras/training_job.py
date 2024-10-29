import abc
import contextlib
import itertools
import os.path as osp
import re
import sys
import threading

import fleras
import fleras.exceptions
import fleras.parallel_map2
import numpy as np
import simplepyutils as spu
import tensorflow as tf
from fleras.optimizers.schedules import WrapperSchedule
from fleras.util.distribute_batch import distribute_batch
from fleras.util.easydict import EasyDict
from simplepyutils import logger


def monkey_patch_tf_delete_file_if_exists():
    def _delete_file_if_exists(filespec):
        """Deletes files matching `filespec`."""
        from tensorflow.python.lib.io import file_io
        from tensorflow.python.framework import errors
        from tensorflow.python.platform import tf_logging as logging

        for pathname in file_io.get_matching_files(filespec):
            try:
                # Truncate the file to free up quota on MLcloud
                file_io.write_string_to_file(pathname, '')
                file_io.delete_file(pathname)
            except errors.NotFoundError:
                logging.warning(
                    "Hit NotFoundError when deleting '%s', possibly because another "
                    "process/thread is also deleting/moving the same file", pathname)

    from tensorflow.python.checkpoint import checkpoint_management
    checkpoint_management._delete_file_if_exists = _delete_file_if_exists


monkey_patch_tf_delete_file_if_exists()


class TrainingJob:
    def __init__(
            self, wandb_project, wandb_config, logdir, init_path, load_path,
            training_steps, stop_step, grad_accum_steps, force_grad_accum, loss_scale,
            dynamic_loss_scale, ema_momentum, finetune_in_inference_mode, validate_period,
            checkpoint_dir, checkpoint_period, multi_gpu, seed, n_completed_steps=None,
            workers=None, parallel_build_data=True):
        # SETTINGS
        self.wandb_project = wandb_project
        self.wandb_config = wandb_config

        self.logdir = logdir

        self.init_path = init_path
        self.load_path = load_path

        self.training_steps = training_steps
        self.stop_step = stop_step
        self.grad_accum_steps = grad_accum_steps
        self.force_grad_accum = force_grad_accum
        self.loss_scale = loss_scale
        self.dynamic_loss_scale = dynamic_loss_scale
        self.ema_momentum = ema_momentum
        self.finetune_in_inference_mode = finetune_in_inference_mode
        self.parallel_build_data = parallel_build_data

        self.validate_period = validate_period

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_period = checkpoint_period

        self.multi_gpu = multi_gpu
        self.seed = seed

        # INITIALIZATION
        self.rng = np.random.Generator(np.random.PCG64(self.seed))
        self.strategy = self._build_distribution_strategy()
        self.n_replicas = self.strategy.num_replicas_in_sync
        if n_completed_steps is not None:
            self._n_completed_substeps_at_start = n_completed_steps * self.grad_accum_steps
        else:
            self._n_completed_substeps_at_start = self._get_n_completed_substeps()

        # These will be filled in while building in self._build()
        self.data_train = None
        self.data_val = None
        self.validation_steps = None
        self.model = None
        self.trainer = None
        self.ckpt_manager = None
        self.callbacks = None

        fleras.parallel_map2.initialize_pool(workers, spu.flags_getter)

    def train(self):
        self._build()
        val_freq = (
            self.validate_period * self.grad_accum_steps if self.validate_period else None)
        suppress_final_checkpoint = False

        logger.info('Starting fitting...')
        try:
            self.trainer.fit_epochless(
                training_data=self.data_train, initial_step=self._n_completed_substeps_at_start,
                steps=self.stop_step * self.grad_accum_steps,
                verbose=1 if sys.stdout.isatty() else 0, callbacks=self.callbacks,
                validation_data=self.data_val,
                validation_freq=val_freq, validation_steps=self.validation_steps)
        except KeyboardInterrupt:
            logger.info('Training interrupted.')
        except tf.errors.ResourceExhaustedError:
            logger.error('Resource Exhausted!')
            # Give a specific return code that may be caught in a shell script
            sys.exit(42)
        except tf.errors.InvalidArgumentError as e:
            # Not saving normal checkpoint as we usually don't want to restore a nonfinite model
            suppress_final_checkpoint = True
            # But saving a separate checkpoint is useful for inspecting what happened
            self.checkpoint.write(
                f'{self.checkpoint_dir}/ckpt_nonfinite_during_training-'
                f'{self.trainer.train_counter.numpy()}')
            logger.info('Saved separate checkpoint.')
            raise
        else:
            # Create SavedModel for the final trained model
            self.model.save(
                f'{self.checkpoint_dir}/model', include_optimizer=False, overwrite=True,
                options=tf.saved_model.SaveOptions(experimental_custom_gradients=True))
            logger.info('Saved final SavedModel.')
        finally:
            if self.trainer.train_counter > 0 and not suppress_final_checkpoint:
                self.ckpt_manager.save(self.trainer.train_counter, check_interval=False)
                logger.info('Saved checkpoint at exit.')

    def _build(self):
        if self.parallel_build_data:
            thread = threading.Thread(target=self._build_data, daemon=True)
            thread.start()
        else:
            self._build_data()

        self._build_trainer()
        self._build_checkpoint_manager()
        self._build_callbacks()
        self._restore_if_ckpt_available()

        if self.parallel_build_data:
            thread.join()

    def _build_data(self):
        logger.info('Building data...')
        data_train, data_val, validation_steps = self.build_data()
        if self.multi_gpu:
            data_train = data_train.prefetch(1)
        else:
            data_train = data_train.apply(tf.data.experimental.prefetch_to_device(
                'GPU:0', 1))
        opt = tf.data.Options()
        opt.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        data_train = data_train.with_options(opt)

        if data_val is not None:
            data_val = data_val.with_options(opt)

        self.data_train = data_train
        self.data_val = data_val
        self.validation_steps = validation_steps
        logger.info('Data built.')

    def _build_trainer(self):
        logger.info('Building trainer...')
        with self.strategy.scope():
            tf.random.set_global_generator(tf.random.Generator.from_seed(self.seed))
            self.model = self.build_model()
            self.trainer = self.build_trainer(self.model)
            self.trainer.gradient_accumulation_steps = self.grad_accum_steps
            optimizer = self._build_optimizer()
            self.trainer.compile(optimizer=optimizer)
            logger.info('Trainer built.')

    def _build_model(self):
        logger.info('Building model...')
        with self.strategy.scope():
            model = self.build_model()
            logger.info('Model built.')
            return model

    def wrap_learning_rate(self, fn):
        return WrapperSchedule(fn, jit_compile=True)

    def _build_optimizer(self):
        kwargs = dict(
            use_ema=self.ema_momentum < 1, ema_momentum=self.ema_momentum, jit_compile=False)

        optimizer = self.build_optimizer(kwargs)

        if self.grad_accum_steps > 1 or self.force_grad_accum:
            optimizer = fleras.optimizers.GradientAccumulationOptimizer(
                optimizer, self.grad_accum_steps, jit_compile=False)

        initial_scale = None if self.loss_scale == 0 else self.loss_scale
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
            optimizer, dynamic=self.dynamic_loss_scale,
            initial_scale=initial_scale)
        optimizer.finalize_variable_values = optimizer.inner_optimizer.finalize_variable_values
        return optimizer

    def _build_callbacks(self):
        cbacks = [
            fleras.callbacks.Checkpoint(self.ckpt_manager),
            fleras.callbacks.ProgbarLogger(self.grad_accum_steps),
            fleras.callbacks.Wandb(
                project_name=self.wandb_project, logdir=self.logdir, config_dict=self.wandb_config,
                grad_accum_steps=self.grad_accum_steps),
        ]
        if self.finetune_in_inference_mode:
            switch_step = (
                    (self.training_steps - self.finetune_in_inference_mode) * self.grad_accum_steps)
            cbacks.append(
                fleras.callbacks.SwitchToInferenceModeCallback(switch_step, self.ckpt_manager))
        self.callbacks = cbacks + list(self.build_callbacks())

    @abc.abstractmethod
    def build_data(self):
        pass

    @abc.abstractmethod
    def build_model(self):
        pass

    @abc.abstractmethod
    def build_trainer(self, model):
        pass

    @abc.abstractmethod
    def build_optimizer(self, default_kwargs):
        pass

    def build_callbacks(self):
        return []

    def _build_checkpoint_manager(self):
        self.checkpoint = tf.train.Checkpoint(model=self.model)
        self.ckpt_manager = tf.train.CheckpointManager(
            self.checkpoint, directory=self.checkpoint_dir, max_to_keep=2,
            step_counter=self.trainer.train_counter, checkpoint_interval=self.checkpoint_period)

    def _build_distribution_strategy(self):
        if self.multi_gpu:
            # return tf.distribute.MultiWorkerMirroredStrategy()
            return tf.distribute.MirroredStrategy(
                #     #cross_device_ops=tf.distribute.ReductionToOneDevice())
                cross_device_ops=tf.distribute.NcclAllReduce())
        else:
            return EasyDict(scope=contextlib.nullcontext, num_replicas_in_sync=1)

    def _get_n_completed_substeps(self):
        load_path = self.get_load_path()
        if load_path is not None and load_path != self.init_path:
            return get_step_count_from_checkpoint_path(load_path)
        else:
            return 0

    def get_load_path(self):
        if self.load_path:
            return self.load_path

        latest = tf.train.latest_checkpoint(self.checkpoint_dir)
        if latest is not None:
            return latest

        return self.init_path

    def _restore_if_ckpt_available(self, expect_partial=False):
        load_path= self.get_load_path()
        if load_path:
            load_path = ensure_absolute_path(load_path, self.checkpoint_dir)
            if load_path.endswith('.index'):
                load_path = osp.splitext(load_path)[0]

            logger.info(f'Restoring from {load_path}...')
            s = self.checkpoint.restore(load_path)
            if expect_partial:
                s.expect_partial()

    # def build_train_flow(
    #         self, examples, load_fn, extra_args, batch_size, roundrobin_sizes=None, rewrap=True):
    #     assert batch_size % self.grad_accum_steps == 0
    #
    #     if roundrobin_sizes is not None and rewrap:
    #         roundrobin_sizes = distribute_batch(
    #             roundrobin_sizes,
    #             batch_size // self.grad_accum_steps,
    #             4 * self.n_replicas * self.grad_accum_steps)
    #
    #     return fleras.parallel_map2.build_dataflow(
    #         examples=examples, load_fn=load_fn, extra_load_fn_args=extra_args,
    #         learning_phase='train', batch_size=batch_size // self.grad_accum_steps,
    #         rng=fleras.parallel_map2.new_rng(self.rng),
    #         n_completed_steps=self._n_completed_substeps_at_start,
    #         roundrobin_sizes=roundrobin_sizes,
    #         n_total_steps=self.training_steps * self.grad_accum_steps)

    def build_stream(self, examples, load_fn, extra_args, shuffle_before_each_epoch=True):
        return _build_stream(
            examples, load_fn, extra_args, shuffle_before_each_epoch,
            rng=fleras.parallel_map2.new_rng(self.rng))

    def build_roundrobin_stream(
            self, example_sections, load_fn, extra_args, batch_size, roundrobin_sizes, rewrap=True,
            shuffle_before_each_epoch=True):
        if rewrap:
            roundrobin_sizes = distribute_batch(
                roundrobin_sizes,
                batch_size // self.grad_accum_steps,
                4 * self.n_replicas * self.grad_accum_steps)
        return _build_roundrobin_stream(
            example_sections, load_fn, extra_args, roundrobin_sizes, shuffle_before_each_epoch,
            rng=fleras.parallel_map2.new_rng(self.rng))

    def merge_streams(self, streams, batch_sizes):
        for b in batch_sizes:
            assert b % self.grad_accum_steps == 0

        return fleras.parallel_map2.roundrobin(
            streams, [b // self.grad_accum_steps for b in batch_sizes])

    def merge_streams_to_tf_dataset_train(self, streams, batch_sizes):
        merged_stream = self.merge_streams(streams, batch_sizes)
        return self.stream_to_tf_dataset_train(merged_stream, sum(batch_sizes))

    def stream_to_tf_dataset_train(self, stream, batch_size):
        return _stream_to_batched_dataset(
            stream, batch_size=batch_size // self.grad_accum_steps,
            n_completed_batches=self._n_completed_substeps_at_start,
            n_total_batches=self.training_steps * self.grad_accum_steps)

    def stream_to_tf_dataset_test(self, stream, batch_size):
        return _stream_to_batched_dataset(
            stream, batch_size=batch_size, n_completed_batches=0, n_total_batches=None)

    def restore(self):
        if self.model is None:
            self.model = self._build_model()
        self.checkpoint = tf.train.Checkpoint(model=self.model)
        self._restore_if_ckpt_available(expect_partial=True)

    def export(self, path):
        path = ensure_absolute_path(path, self.checkpoint_dir)
        self.model.save(
            path, include_optimizer=False, overwrite=True,
            options=tf.saved_model.SaveOptions(experimental_custom_gradients=True))
        logger.info('Exported SavedModel.')


def _stream_to_batched_dataset(stream, batch_size, n_completed_batches, n_total_batches=None):
    n_completed_items = n_completed_batches * batch_size
    n_total_items = n_total_batches * batch_size if n_total_batches is not None else None
    sliced_stream = itertools.islice(stream, n_completed_items, n_total_items)
    n_remaining_batches = n_total_batches - n_completed_batches if n_total_batches else None
    return fleras.parallel_map2.function_calls_to_batched_tf_dataset(
        sliced_stream, batch_size, n_remaining_batches)


def _build_stream(examples, load_fn, extra_args, shuffle_before_each_epoch, rng):
    shuffler_rng = fleras.parallel_map2.new_rng(rng)
    loader_rng = fleras.parallel_map2.new_rng(rng)
    item_stream = fleras.parallel_map2.iterate_repeatedly(
        examples, shuffle_before_each_epoch, shuffler_rng)
    return _build_fns_args_kwargs_stream(item_stream, load_fn, extra_args, rng=loader_rng)


def _build_roundrobin_stream(
        example_sections, load_fn, extra_args, roundrobin_sizes,
        shuffle_before_each_epoch, rng):
    item_streams = [
        fleras.parallel_map2.iterate_repeatedly(
            examples, shuffle_before_each_epoch, fleras.parallel_map2.new_rng(rng))
        for examples in example_sections]

    fns_args_kwargs_streams = [
        _build_fns_args_kwargs_stream(
            item_stream, load_fn, extra_args, rng=fleras.parallel_map2.new_rng(rng))
        for item_stream in item_streams]

    return fleras.parallel_map2.roundrobin(fns_args_kwargs_streams, roundrobin_sizes)


def _build_fns_args_kwargs_stream(items, load_fn, extra_args, rng):
    for item in items:
        yield load_fn, (item, *extra_args), dict(rng=fleras.parallel_map2.new_rng(rng))


def get_step_count_from_checkpoint_path(checkpoint_path):
    return int(re.search(r'ckpt-(?P<num>\d+)', checkpoint_path)['num'])


def ensure_absolute_path(path, root):
    if not root:
        return path

    if osp.isabs(path):
        return path
    else:
        return osp.join(root, path)
