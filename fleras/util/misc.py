from fleras.util.easydict import EasyDict

def attrdict2dict_nested(elem):
    if isinstance(elem, dict):
        return {k: attrdict2dict_nested(v) for k, v in elem.items()}
    elif isinstance(elem, tuple):
        return tuple([attrdict2dict_nested(e) for e in elem])
    elif isinstance(elem, list):
        return [attrdict2dict_nested(e) for e in elem]
    else:
        return elem


def dict2attrdict_nested(elem):
    #def factory():
    #    return attrdict.AttrDefault(default_factory=factory)

    if isinstance(elem, dict):
        return EasyDict({k: dict2attrdict_nested(v) for k, v in elem.items()})
        #return attrdict.AttrDefault(
        #    default_factory=factory,
        #    items={k: dict2attrdict_nested(v) for k, v in elem.items()})
    elif isinstance(elem, tuple):
        return tuple([dict2attrdict_nested(e) for e in elem])
    elif isinstance(elem, list):
        return [dict2attrdict_nested(e) for e in elem]
    else:
        return elem


def sync_to_numpy_or_python_type(tensors):
    # Copied from keras.utils.tf_utils.sync_to_numpy_or_python_type
    """Syncs and converts a structure of `Tensor`s to `NumPy` arrays or Python
    scalar types.

    For each tensor, it calls `tensor.numpy()`. If the result is a scalar value,
    it converts it to a Python type, such as a float or int, by calling
    `result.item()`.

    Numpy scalars are converted, as Python types are often more convenient to
    deal with. This is especially useful for bfloat16 Numpy scalars, which don't
    support as many operations as other Numpy values.

    Async strategies (such as `TPUStrategy` and `ParameterServerStrategy`) are
    forced to
    sync during this process.

    Args:
      tensors: A structure of tensors.

    Returns:
      `tensors`, but scalar tensors are converted to Python types and non-scalar
      tensors are converted to Numpy arrays.
    """
    if isinstance(tensors, tf.distribute.experimental.coordinator.RemoteValue):
        tensors = tensors.fetch()

    def _to_single_numpy_or_python_type(t):
        # Don't turn ragged or sparse tensors to NumPy.
        if isinstance(t, tf.Tensor):
            t = t.numpy()
        # Strings, ragged and sparse tensors don't have .item(). Return them
        # as-is.
        if not isinstance(t, (np.ndarray, np.generic)):
            return t
        return t.item() if np.ndim(t) == 0 else t

    return tf.nest.map_structure(_to_single_numpy_or_python_type, tensors)