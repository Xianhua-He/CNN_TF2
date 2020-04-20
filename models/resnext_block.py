import tensorflow as tf
from models.group_convolution import GroupConv2D
from tensorflow.keras import layers


class ResNeXt_BottleNeck(tf.keras.layers.Layer):
    def __init__(self, filters, strides, groups):
        super(ResNeXt_BottleNeck, self).__init__()
        self.conv1 = layers.Conv2D(filters, 1, 1, "same")
        self.bn1 = layers.BatchNormalization()
        self.group_conv = GroupConv2D(filters, filters, 3, strides, "same", groups)
        self.bn2 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(2*filters, 1, 1, "same")
        self.bn3 = layers.BatchNormalization()
        self.shortcut_conv = layers.Conv2D(2*filters, 1, strides, "same")
        self.shortcut_bn = layers.BatchNormalization()

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.group_conv(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn3(x, training=training)
        x = tf.nn.relu(x)

        shortcut = self.shortcut_conv(inputs)
        shortcut = self.shortcut_bn(shortcut, training=training)

        output = tf.nn.relu(layers.add([x, shortcut]))
        return output


def build_ResNeXt_block(filters, strides, groups, repeat_num):
    block = tf.keras.Sequential()
    block.add(ResNeXt_BottleNeck(filters, strides, groups))
    for _ in range(1, repeat_num):
        block.add(ResNeXt_BottleNeck(filters, 1, groups))

    return block