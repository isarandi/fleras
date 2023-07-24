import tensorflow as tf


def wrap(jit_compile=False):
    def wrapper(fn):
        def wrapped():
            return WrapperSchedule(fn, jit_compile=jit_compile)

        return wrapped

    return wrapper


class WrapperSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, function, jit_compile=False):
        self.function = function
        if jit_compile:
            self.function = tf.function(self.function)

    def __call__(self, step):
        return self.function(step)
