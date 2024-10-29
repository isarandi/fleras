import atexit
import ctypes
import itertools
import multiprocessing as mp
import os
import queue
import signal
import threading

import more_itertools
import numpy as np
import tensorflow as tf

_pool = None


def initialize_pool(n_workers=None, flag_namespace_getter=None):
    if n_workers is None:
        n_workers = min(len(os.sched_getaffinity(0)), 12)

    # important to use 'spawn', because 'fork' would mean the whole memory is (lazily) copied
    # then due to copy-on-write semantics, it gets duplicated when the parent changes anything
    ctx = mp.get_context('spawn')
    global _pool

    if flag_namespace_getter is None:
        _pool = ctx.Pool(n_workers, initializer=init_worker_process)
    else:
        flag_values = flag_namespace_getter()
        _pool = ctx.Pool(
            n_workers, initializer=_init_worker_process_with_flags,
            initargs=(flag_values, flag_namespace_getter,))
    return _pool


def function_calls_to_batched_tf_dataset(fns_args_kwargs, batch_size, n_batches=None):
    processed_items_genfunc = parallel_map_as_generator_raw(fns_args_kwargs, batch_size)
    return iterable_to_batched_tf_dataset(processed_items_genfunc(), batch_size, n_batches)


def iterable_to_batched_tf_dataset(iterable, batch_size, n_batches=None, drop_remainder=False):
    processed_batches = more_itertools.chunked(iterable, batch_size)
    merged_batches = (merge(batch) for batch in processed_batches)
    return iterable_to_tf_dataset(
        merged_batches, n_items=n_batches, unknown_first_dim=not drop_remainder)


def parallel_map_as_generator_raw(fns_args_kwargs, max_unconsumed=256):
    semaphore = threading.Semaphore(max_unconsumed)
    q = queue.Queue()
    end_of_sequence_marker = object()
    should_stop = False

    if _pool is None:
        raise RuntimeError("Pool not initialized. Call `initialize_pool` first.")

    def producer():
        for fn, args, kwargs in fns_args_kwargs:
            if should_stop:
                break
            semaphore.acquire()
            q.put(_pool.apply_async(fn, args, kwargs))

        q.put(end_of_sequence_marker)

    def consumer():
        while (future := q.get()) is not end_of_sequence_marker:
            value = future.get()
            semaphore.release()
            yield value

    def stop():
        nonlocal should_stop
        should_stop = True
        _pool.close()
        _pool.terminate()

    producer_thread = threading.Thread(target=producer, daemon=True)
    producer_thread.start()
    atexit.register(stop)

    return consumer


def merge(batch):
    result = {}
    for element in batch:
        for key, value in element.items():
            append_nested(result, key, value)
    return stack_nested(key=None, value=result)


def append_nested(result_tree, key, value_tree):
    if isinstance(value_tree, dict):
        for subkey, subvalue in value_tree.items():
            if key not in result_tree:
                result_tree[key] = {}
            append_nested(result_tree[key], subkey, subvalue)
    # if numpy array or scalar
    elif isinstance(value_tree, np.ndarray) or np.isscalar(value_tree):
        if key not in result_tree:
            result_tree[key] = []
        result_tree[key].append(value_tree)
    else:
        raise RuntimeError(f'Unexpected type {type(value_tree)}')


def stack_nested(key, value):
    if isinstance(value, dict):
        return {
            subkey: stack_nested(subkey, subvalue) for subkey, subvalue in value.items()}
    elif isinstance(value, list):
        if key is not None and key.startswith('_ragged_'):
            # Since numpy has no ragged array, we represent it as a dict.
            # It will be constructed into tf.RaggedTensor in `iterable_to_tf_dataset`
            # via `compose_raggeds`.
            return dict(
                values=np.concatenate(value, axis=0),
                row_lengths=np.array([len(x) for x in value], np.int32))
        # elif key is not None and key.startswith('_sparse_'):
        #     stacked = scipy.sparse.vstack(value).tocoo()
        #     return dict(
        #         values=stacked.data,
        #         indices=np.stack([stacked.row, stacked.col], axis=1),
        #         dense_shape=stacked.shape)
        else:
            return np.stack(value)
    else:
        raise RuntimeError(f'Unexpected type {type(value)}')


def iterable_to_tf_dataset(iterable, n_items=None, unknown_first_dim=False):
    (first_elem,), iterable = more_itertools.spy(iterable)
    output_signature = tf.nest.map_structure(tf.type_spec_from_value, first_elem)
    if unknown_first_dim:
        output_signature = remove_first_dim_shape_info(key=None, value=output_signature)

    output_signature = remove_ragged_value_shape_info(key=None, value=output_signature)
    ds = tf.data.Dataset.from_generator(lambda: iterable, output_signature=output_signature)
    ds = ds.map(compose_ragged_and_sparse)

    # Make the cardinality of the dataset known to TF.
    if n_items is not None:
        ds = ds.take(n_items)
    return ds


def remove_ragged_value_shape_info(key, value):
    if isinstance(value, dict):
        if key is not None and key.startswith('_ragged_'):
            value['values'] = tf.TensorSpec(
                dtype=value['values'].dtype, shape=[None, *value['values'].shape[1:]])
            return value
        else:
            return {k: remove_ragged_value_shape_info(k, v) for k, v in value.items()}
    else:
        return value


def remove_first_dim_shape_info(key, value):
    if isinstance(value, dict):
        return {k: remove_first_dim_shape_info(k, v) for k, v in value.items()}
    else:
        return tf.TensorSpec(dtype=value.dtype, shape=[None, *value.shape[1:]])


def compose_ragged_and_sparse(value):
    return _compose_ragged_and_sparse(key=None, value=value)


def _compose_ragged_and_sparse(key, value):
    if isinstance(value, dict):
        if key is not None and key.startswith('_ragged_'):
            return tf.RaggedTensor.from_row_lengths(value['values'], value['row_lengths'])
        #         elif key is not None and key.startswith('_sparse_'):
        # #            return tf.SparseTensor(
        # #                indices=tf.cast(value['indices'], tf.int64),
        # #                values=value['values'],
        # #                dense_shape=tf.cast(value['dense_shape'], tf.int64))
        #             return tf.raw_ops.SparseTensorToCSRSparseMatrix(
        #                 indices=tf.cast(value['indices'], tf.int64),
        #                 values=value['values'],
        #                 dense_shape=tf.cast(value['dense_shape'], tf.int64))
        else:
            return {
                subkey.removeprefix('_ragged_').removeprefix('_sparse_'):
                    _compose_ragged_and_sparse(subkey, subvalue)
                for subkey, subvalue in value.items()}
    else:
        return value


def _init_worker_process_with_flags(flag_values, flag_namespace_getter):
    flags_target = flag_namespace_getter()
    for key in flag_values.__dict__:
        setattr(flags_target, key, getattr(flag_values, key))
    init_worker_process()


def init_worker_process():
    import numpy as np
    np.bool = bool
    np.complex = complex
    np.str = str
    _terminate_on_parent_death()
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _terminate_on_parent_death():
    prctl = ctypes.CDLL("libc.so.6").prctl
    PR_SET_PDEATHSIG = 1
    prctl(PR_SET_PDEATHSIG, signal.SIGTERM)


MAX_INT = 2 ** 32 - 1


def new_rng(rng):
    return np.random.Generator(np.random.PCG64(rng.integers(0, MAX_INT)))


def iterate_repeatedly(seq, shuffle_before_each_epoch=False, rng=None):
    """Iterates over and yields the elements of `iterable` over and over.
    If `shuffle_before_each_epoch` is True, the elements are put in a list and shuffled before
    every pass over the data, including the first."""

    if rng is None:
        rng = np.random.default_rng()

    # create a (shallow) copy so shuffling only applies to the copy.
    seq = list(seq)
    rng.shuffle(seq)
    yield from seq

    while True:
        if shuffle_before_each_epoch:
            rng.shuffle(seq)
        yield from seq


def roundrobin(iterables, sizes):
    iterators = [iter(iterable) for iterable in iterables]
    for iterator, size in zip(itertools.cycle(iterators), itertools.cycle(sizes)):
        for _ in range(size):
            try:
                yield next(iterator)
            except StopIteration:
                return
