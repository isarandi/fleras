import tensorflow as tf


class TrainTestSwitchLayer(tf.keras.layers.Layer):
    def __init__(self, train_layer, test_layer):
        super(TrainTestSwitchLayer, self).__init__()
        self.train_layer = train_layer
        self.test_layer = test_layer
        self.input_spec = self.train_layer.input_spec

    def build(self, input_shape):
        self.train_layer.build(input_shape)
        self.test_layer.build(input_shape)

    def call(self, inputs, *args, training=None, **kwargs):
        layer = self.train_layer if training else self.test_layer
        return layer(inputs, *args, training=training, **kwargs)
