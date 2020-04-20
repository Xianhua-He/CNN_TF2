import tensorflow as tf
from configuration import NUM_CLASSES
from tensorflow.keras import layers

class BottleNeck(tf.keras.layers.Layer):
    def __init__(self, growth_rate, drop_rate):
        super(BottleNeck, self).__init__()
        self.bn1 = layers.BatchNormalization()
        self.conv1 = layers.Conv2D(filters=4*growth_rate, kernel_size=1, strides=1, padding="same")
        self.bn2 = layers.BatchNormalization()
        self.conv2= layers.Conv2D(growth_rate, 3, 1, "same")
        self.dropout = layers.Dropout(rate=drop_rate)

    def call(self, inputs, training=None, **kwargs):
        x = self.bn1(inputs, training=training)
        x = tf.nn.relu(x)

        x = self.conv1(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.dropout(x, training=training)
        return x


class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, num_layers, growth_rate, drop_rate):
        super(DenseBlock, self).__init__()
        self.num_layers = num_layers
        self.growth_rate = growth_rate
        self.drop_rate = drop_rate
        self.features_list = []

    def _make_layer(self, x, training=None):
        y = BottleNeck(growth_rate=self.growth_rate, drop_rate=self.drop_rate)(x, training=training)
        self.features_list.append(y)
        y = tf.concat(self.features_list, axis=-1)
        return y

    def call(self, inputs, training=None, **kwargs):
        self.features_list.append(inputs)
        x = self._make_layer(inputs, training=training)
        for i in range(1, self.num_layers):
            x = self._make_layer(x, training=training)
        self.features_list.clear()
        return x


class TransitionLayer(tf.keras.layers.Layer):
    def __init__(self, out_channels):
        super(TransitionLayer, self).__init__()
        self.bn = layers.BatchNormalization()
        self.conv = layers.Conv2D(filters=out_channels, kernel_size=(1, 1), strides=1, padding="same")
        self.pool = layers.MaxPool2D(2, 2, "same")

    def call(self, inputs, training=None, **kwargs):
        x = self.bn(inputs, training=training)
        x = tf.nn.relu(x)
        x = self.conv(x)
        x = self.pool(x)
        return x


class DenseNet(tf.keras.Model):
    def __init__(self, num_init_features, growth_rate, block_layers, compression_rate, drop_rate):
        super(DenseNet, self).__init__()
        self.conv = layers.Conv2D(num_init_features, 7, 2, "same")
        self.bn = layers.BatchNormalization()
        self.pool = layers.MaxPool2D(3, 2, "same")
        self.num_channels = num_init_features
        self.dense_block_1 = DenseBlock(num_layers=block_layers[0], growth_rate=growth_rate, drop_rate=drop_rate)
        self.num_channels += growth_rate * block_layers[0]
        self.num_channels = compression_rate * self.num_channels
        self.transition_1 = TransitionLayer(out_channels=int(self.num_channels))

        self.dense_block_2 = DenseBlock(num_layers=block_layers[1], growth_rate=growth_rate, drop_rate=drop_rate)
        self.num_channels += growth_rate * block_layers[1]
        self.num_channels = compression_rate * self.num_channels
        self.transition_2 = TransitionLayer(out_channels=int(self.num_channels))

        self.dense_block_3 = DenseBlock(block_layers[2], growth_rate, drop_rate)
        self.num_channels += growth_rate * block_layers[2]
        self.num_channels = compression_rate * self.num_channels
        self.transition_3 = TransitionLayer(out_channels=int(self.num_channels))

        self.dense_block_4 = DenseBlock(block_layers[3], growth_rate, drop_rate)

        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(NUM_CLASSES, activation=tf.keras.activations.softmax)

    def call(self, inputs, training=None, mask=None):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool(x)

        x = self.dense_block_1(x, training=training)
        x = self.transition_1(x, training=training)
        x = self.dense_block_2(x, training=training)
        x = self.transition_2(x, training=training)
        x = self.dense_block_3(x, training=training)
        x = self.transition_3(x, training=training)
        x = self.dense_block_4(x, training=training)

        x = self.avgpool(x)
        x = self.fc(x)
        return x


def densenet_121():
    return DenseNet(num_init_features=64, growth_rate=32, block_layers=[6, 12, 24, 16], compression_rate=0.5, drop_rate=0.5)

def densenet_169():
    return DenseNet(num_init_features=64, growth_rate=32, block_layers=[6, 12, 32, 32], compression_rate=0.5, drop_rate=0.5)


def densenet_201():
    return DenseNet(num_init_features=64, growth_rate=32, block_layers=[6, 12, 48, 32], compression_rate=0.5, drop_rate=0.5)


def densenet_264():
    return DenseNet(num_init_features=64, growth_rate=32, block_layers=[6, 12, 64, 48], compression_rate=0.5, drop_rate=0.5)

