import tensorflow as tf
from tensorflow.keras import layers
from configuration import NUM_CLASSES


class MobileNetV1(tf.keras.Model):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.conv1 = layers.Conv2D(32, 3, 2, "same")
        self.separable_conv_1 = layers.SeparableConv2D(64, 3, 1, "same", activation=tf.nn.relu6)

        self.separable_conv_2 = layers.SeparableConv2D(128, 3, 2, "same", activation=tf.nn.relu6)
        self.separable_conv_3 = layers.SeparableConv2D(128, 3, 1, "same", activation=tf.nn.relu6)

        self.separable_conv_4 = layers.SeparableConv2D(256, 3, 2, "same", activation=tf.nn.relu6)
        self.separable_conv_5 = layers.SeparableConv2D(256, 3, 1, "same", activation=tf.nn.relu6)

        self.separable_conv_6 = layers.SeparableConv2D(512, 3, 2, "same", activation=tf.nn.relu6)

        # 5* (512, 3, 1, "same")
        self.separable_conv_7 = layers.SeparableConv2D(512, 3, 1, "same", activation=tf.nn.relu6)
        self.separable_conv_8 = layers.SeparableConv2D(512, 3, 1, "same", activation=tf.nn.relu6)
        self.separable_conv_9 = layers.SeparableConv2D(512, 3, 1, "same", activation=tf.nn.relu6)
        self.separable_conv_10 = layers.SeparableConv2D(512, 3, 1, "same", activation=tf.nn.relu6)
        self.separable_conv_11 = layers.SeparableConv2D(512, 3, 1, "same", activation=tf.nn.relu6)

        self.separable_conv_12 = layers.SeparableConv2D(1024, 3, 2, "same", activation=tf.nn.relu6)
        self.separable_conv_13 = layers.SeparableConv2D(1024, 3, 2, "same", activation=tf.nn.relu6)

        self.avg_pool = layers.AveragePooling2D((7, 7))
        self.fc = layers.Dense(NUM_CLASSES, activation=tf.keras.activations.softmax)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.separable_conv_1(x)
        x = self.separable_conv_2(x)
        x = self.separable_conv_3(x)
        x = self.separable_conv_4(x)
        x = self.separable_conv_5(x)
        x = self.separable_conv_6(x)
        x = self.separable_conv_7(x)
        x = self.separable_conv_8(x)
        x = self.separable_conv_9(x)
        x = self.separable_conv_10(x)
        x = self.separable_conv_11(x)
        x = self.separable_conv_12(x)
        x = self.separable_conv_13(x)

        x = self.avg_pool(x)
        x = self.fc(x)

        return x
