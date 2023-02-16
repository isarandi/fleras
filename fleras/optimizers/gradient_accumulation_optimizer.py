import tensorflow as tf


class GradientAccumulationOptimizer(tf.keras.optimizers.Optimizer):
    def __init__(self, inner_optimizer, num_steps, name='GradientAccumulationOptimizer', **kwargs):
        super().__init__(name, **kwargs)
        self.inner_optimizer = inner_optimizer
        self.num_steps = num_steps
        self._learning_rate = None

        # Patch the inner update step
        if self.num_steps > 1:
            self.old_inner_update_step = self.inner_optimizer.update_step
            self.inner_optimizer.update_step = self.new_inner_update_step

    def build(self, var_list):
        super().build(var_list)
        if hasattr(self, '_built') and self._built:
            return
        self._grad_accums = [
            self.add_variable_from_reference(var, 'grad_accum') for var in var_list]

        self._built = True

    def apply_gradients(self, grads_and_vars, **kwargs):
        if self.num_steps == 1:
            return self.inner_optimizer.apply_gradients(grads_and_vars, **kwargs)

        # This will accumulate the gradient into the accumulator (potentially resetting it before)
        super().apply_gradients(grads_and_vars, **kwargs)

        # Get the accumulated gradients
        accum_grads_and_vars = [
            (self._grad_accums[self._index_dict[self._var_key(v)]], v)
            for g, v in grads_and_vars]

        # Apply gradients, but since we patched the inner optimizer in the constructor,
        # it will only update variables if it's time to do it (every num_steps steps)
        self.inner_optimizer.iterations.assign((self.iterations - 1) // self.num_steps)
        self.inner_optimizer.apply_gradients(accum_grads_and_vars, **kwargs)


    def update_step(self, gradient, variable):
        var_key = self._var_key(variable)
        grad_accum = self._grad_accums[self._index_dict[var_key]]

        if isinstance(gradient, tf.IndexedSlices):
            def starting():
                grad_accum.assign(tf.zeros_like(grad_accum))
                grad_accum.scatter_add(gradient)

            def intermediate():
                grad_accum.scatter_add(gradient)
        else:
            def starting():
                grad_accum.assign(gradient)

            def intermediate():
                grad_accum.assign_add(gradient)

        tf.cond(self.iterations % self.num_steps == 0, starting, intermediate)

    def new_inner_update_step(self, gradient, variable):
        def do_inner_update():
            n_steps = tf.cast(self.num_steps, tf.float32)
            self.old_inner_update_step(gradient / n_steps, variable)

        return tf.cond(self.iterations % self.num_steps == 0, do_inner_update, lambda: None)

    def finalize_variable_values(self, variables):
        self.inner_optimizer.finalize_variable_values(variables)

    @property
    def learning_rate(self):
        return self.inner_optimizer.learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self.inner_optimizer.learning_rate = learning_rate
