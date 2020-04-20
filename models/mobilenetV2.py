import tensorflow as tf
from tensorflow.keras import layers
from configuration import NUM_CLASSES


class BottleNeck(tf.keras.layers.Layer):
    def __init__(self, input_channels, output_channels, expansion_factor, stride):
        super(BottleNeck, self).__init__()
        self.stride = stride
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.conv1 = layers.Conv2D(input_channels*expansion_factor, 1, 1, "same")
        self.bn1 = layers.BatchNormalization()
        self.dwconv = layers.DepthwiseConv2D(3, stride, "same")
        self.bn2 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(output_channels, 1, 1, "same")
        self.bn3 = layers.BatchNormalization()
        self.linear = layers.Activation(tf.keras.activations.linear)

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu6(x)
        x = self.dwconv(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu6(x)
        x = self.conv2(x)
        x = self.bn3(x, training=training)
        x = self.linear(x)
        if self.stride == 1 and self.input_channels == self.output_channels:
            x = layers.add([x, inputs])
        return x


def build_bottleneck(t, in_channel_num, out_channel_num, n, s):
    # t: expansion factor
    # s: stride
    # n : repeat times
    bottleneck = tf.keras.Sequential()
    for i in range(n):
        if i == 0:
            bottleneck.add(BottleNeck(in_channel_num, out_channel_num, t, s))
        else:
            bottleneck.add(BottleNeck(out_channel_num, out_channel_num, t, 1))

    return bottleneck


class MobileNetV2(tf.keras.Model):
    def __init__(self):
        super(MobileNetV2, self).__init__()
        self.conv1 = layers.Conv2D(32, 3, 2, "same")
        self.bottleneck_1 = build_bottleneck(t=1, in_channel_num=32, out_channel_num=64, n=1, s=1)
        self.bottleneck_2 = build_bottleneck(6, 16, 24, 2, 2)
        self.bottleneck_3 = build_bottleneck(t=6,
                                             in_channel_num=24,
                                             out_channel_num=32,
                                             n=3,
                                             s=2)
        self.bottleneck_4 = build_bottleneck(t=6,
                                             in_channel_num=32,
                                             out_channel_num=64,
                                             n=4,
                                             s=2)
        self.bottleneck_5 = build_bottleneck(t=6,
                                             in_channel_num=64,
                                             out_channel_num=96,
                                             n=3,
                                             s=1)
        self.bottleneck_6 = build_bottleneck(t=6,
                                             in_channel_num=96,
                                             out_channel_num=160,
                                             n=3,
                                             s=2)
        self.bottleneck_7 = build_bottleneck(t=6,
                                             in_channel_num=160,
                                             out_channel_num=320,
                                             n=1,
                                             s=1)
        self.conv2 = layers.Conv2D(1280, 1, 1, "same")
        self.avgpool = layers.AveragePooling2D((7, 7))
        self.conv3 = layers.Conv2D(NUM_CLASSES, 1, 1, "same", activation=tf.keras.activations.softmax)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bottleneck_1(x, training=training)
        x = self.bottleneck_2(x, training=training)
        x = self.bottleneck_3(x, training=training)
        x = self.bottleneck_4(x, training=training)
        x = self.bottleneck_5(x, training=training)
        x = self.bottleneck_6(x, training=training)
        x = self.bottleneck_7(x, training=training)

        x = self.conv2(x)
        x = self.avgpool(x)
        x = self.conv3(x)

        return x






