import tensorflow as tf
from tensorflow.keras import layers


class BasicConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=(1, 1), strides=1, padding="same"):
        super(BasicConv2D, self).__init__()
        self.conv = layers.Conv2D(filters, kernel_size, strides, padding)
        self.bn = layers.BatchNormalization()

    def call(self, inputs, training=None, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        return x


class Conv2DLinear(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=(1, 1), strides=1, padding="same"):
        super(Conv2DLinear, self).__init__()
        self.conv = layers.Conv2D(filters, kernel_size, strides, padding)
        self.bn = layers.BatchNormalization()

    def call(self, inputs, training=None, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        return x


class Stem(tf.keras.layers.Layer):
    def __init__(self):
        super(Stem, self).__init__()
        self.conv1 = BasicConv2D(32, (3, 3), 2, "valid")
        self.conv2 = BasicConv2D(32, 3, 1, "valid")
        self.conv3 = BasicConv2D(64, (3, 3), 1, "same")
        self.b1_maxpool = layers.MaxPool2D(3, 2, "valid")
        self.b2_conv = BasicConv2D(96, (3, 3), 2, "valid")
        self.b3_conv1 = BasicConv2D(64, (1, 1), 1, "same")
        self.b3_conv2 = BasicConv2D(96, (3, 3), 1, "valid")
        self.b4_conv1 = BasicConv2D(64, (1, 1), 1, "same")
        self.b4_conv2 = BasicConv2D(64, (7, 1), 1, "same")
        self.b4_conv3 = BasicConv2D(64, (1, 7), 1, "same")
        self.b4_conv4 = BasicConv2D(96, (3, 3), 1, "valid")
        self.b5_conv = BasicConv2D(192, (3, 3), 2, "valid")
        self.b6_maxpool = layers.MaxPool2D(3, 2, "valid")

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        branch_1 = self.b1_maxpool(x)
        branch_2 = self.b2_conv(x, training=training)
        x = tf.concat(values=[branch_1, branch_2], axis=-1)
        branch_3 = self.b3_conv1(x, training=training)
        branch_3 = self.b3_conv2(branch_3, training=training)
        branch_4 = self.b4_conv1(x, training=training)
        branch_4 = self.b4_conv2(branch_4, training=training)
        branch_4 = self.b4_conv3(branch_4, training=training)
        branch_4 = self.b4_conv4(branch_4, training=training)
        x = tf.concat(values=[branch_3, branch_4], axis=-1)
        branch_5 = self.b5_conv(x, training=training)
        branch_6 = self.b6_maxpool(x)
        x = tf.concat([branch_5, branch_6], axis=-1)
        return x


class InceptionBlockA(tf.keras.layers.Layer):
    def __init__(self):
        super(InceptionBlockA, self).__init__()
        self.b1_pool = layers.AveragePooling2D((3, 3), 1, "same")
        self.b1_conv = BasicConv2D(96, (1, 1), 1, "same")
        self.b2_conv = BasicConv2D(96, 1, 1, "same")
        self.b3_conv1 = BasicConv2D(64, 1, 1, "same")
        self.b3_conv2 = BasicConv2D(96, 3, 1, "same")
        self.b4_conv1 = BasicConv2D(64, 1, 1, "same")
        self.b4_conv2 = BasicConv2D(96, 3, 1, "same")
        self.b4_conv3 = BasicConv2D(94, 3, 1, "same")

    def call(self, inputs, training=None, **kwargs):
        b1 = self.b1_pool(inputs)
        b1 = self.b1_conv(b1, training=training)

        b2 = self.b2_conv(inputs, training=training)

        b3 = self.b3_conv1(inputs, training=training)
        b3 = self.b3_conv2(b3, training=training)

        b4 = self.b4_conv1(inputs, training=training)
        b4 = self.b4_conv2(b4, training=training)
        b4 = self.b4_conv3(b4, training=training)

        return tf.concat(values=[b1, b2, b3, b4], axis=-1)


class ReductionA(tf.keras.layers.Layer):
    def __init__(self, n, k, l, m):
        self.b1_pool = layers.MaxPool2D(3, 2, "valid")
        self.b2_conv = BasicConv2D(n, (3, 3), 2, "valid")
        self.b3_conv1 = BasicConv2D(k, (1, 1), 1, "same")
        self.b3_conv2 = BasicConv2D(l, (3, 3), 1, "same")
        self.b3_conv3 = BasicConv2D(m, (3, 3), 2, "valid")

    def call(self, inputs, training=None, **kwargs):
        b1 = self.b1_pool(inputs)

        b2 = self.b2_conv(inputs, training=training)

        b3 = self.b3_conv1(inputs, training=training)
        b3 = self.b3_conv2(b3, training=training)
        b3 = self.b3_conv3(b3, training=training)

        return tf.concat([b1, b2, b3], axis=-1)


class InceptionBlockB(tf.keras.layers.Layer):
    def __init__(self):
        super(InceptionBlockB, self).__init__()
        self.b1_pool = layers.AveragePooling2D((3, 3), 1, "same")
        self.b1_conv = BasicConv2D(128, (1, 1), 1, "same")
        self.b2_conv = BasicConv2D(filters=384,
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding="same")
        self.b3_conv1 = BasicConv2D(filters=192,
                                    kernel_size=(1, 1),
                                    strides=1,
                                    padding="same")
        self.b3_conv2 = BasicConv2D(filters=224,
                                    kernel_size=(1, 7),
                                    strides=1,
                                    padding="same")
        self.b3_conv3 = BasicConv2D(filters=256,
                                    kernel_size=(1, 7),
                                    strides=1,
                                    padding="same")
        self.b4_conv1 = BasicConv2D(filters=192,
                                    kernel_size=(1, 1),
                                    strides=1,
                                    padding="same")
        self.b4_conv2 = BasicConv2D(filters=192,
                                    kernel_size=(1, 7),
                                    strides=1,
                                    padding="same")
        self.b4_conv3 = BasicConv2D(filters=224,
                                    kernel_size=(7, 1),
                                    strides=1,
                                    padding="same")
        self.b4_conv4 = BasicConv2D(filters=224,
                                    kernel_size=(1, 7),
                                    strides=1,
                                    padding="same")
        self.b4_conv5 = BasicConv2D(filters=256,
                                    kernel_size=(7, 1),
                                    strides=1,
                                    padding="same")


    def call(self, inputs, training=None, **kwargs):
        b1 = self.b1_pool(inputs)
        b1 = self.b1_conv(b1, training=training)

        b2 = self.b2_conv(inputs, training=training)

        b3 = self.b3_conv1(inputs, training=training)
        b3 = self.b3_conv2(b3, training=training)
        b3 = self.b3_conv3(b3, training=training)

        b4 = self.b4_conv1(inputs, training=training)
        b4 = self.b4_conv2(b4, training=training)
        b4 = self.b4_conv3(b4, training=training)
        b4 = self.b4_conv4(b4, training=training)
        b4 = self.b4_conv5(b4, training=training)

        return tf.concat(values=[b1, b2, b3, b4], axis=-1)


class ReductionB(tf.keras.layers.Layer):
    def __init__(self):
        super(ReductionB, self).__init__()
        self.b1_pool = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                                 strides=2,
                                                 padding="valid")
        self.b2_conv1 = BasicConv2D(filters=192,
                                    kernel_size=(1, 1),
                                    strides=1,
                                    padding="same")
        self.b2_conv2 = BasicConv2D(filters=192,
                                    kernel_size=(3, 3),
                                    strides=2,
                                    padding="valid")
        self.b3_conv1 = BasicConv2D(filters=256,
                                    kernel_size=(1, 1),
                                    strides=1,
                                    padding="same")
        self.b3_conv2 = BasicConv2D(filters=256,
                                    kernel_size=(1, 7),
                                    strides=1,
                                    padding="same")
        self.b3_conv3 = BasicConv2D(filters=320,
                                    kernel_size=(7, 1),
                                    strides=1,
                                    padding="same")
        self.b3_conv4 = BasicConv2D(filters=320,
                                    kernel_size=(3, 3),
                                    strides=2,
                                    padding="valid")

    def call(self, inputs, training=None, **kwargs):
        b1 = self.b1_pool(inputs)

        b2 = self.b2_conv1(inputs, training=training)
        b2 = self.b2_conv2(b2, training=training)

        b3 = self.b3_conv1(inputs, training=training)
        b3 = self.b3_conv2(b3, training=training)
        b3 = self.b3_conv3(b3, training=training)
        b3 = self.b3_conv4(b3, training=training)

        return tf.concat(values=[b1, b2, b3], axis=-1)


class InceptionBlockC(tf.keras.layers.Layer):
    def __init__(self):
        super(InceptionBlockC, self).__init__()
        self.b1_pool = tf.keras.layers.AveragePooling2D(pool_size=(3, 3),
                                                        strides=1,
                                                        padding="same")
        self.b1_conv = BasicConv2D(filters=256,
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding="same")
        self.b2_conv = BasicConv2D(filters=256,
                                   kernel_size=(1, 1),
                                   strides=1,
                                   padding="same")
        self.b3_conv1 = BasicConv2D(filters=384,
                                    kernel_size=(1, 1),
                                    strides=1,
                                    padding="same")
        self.b3_conv2 = BasicConv2D(filters=256,
                                    kernel_size=(1, 3),
                                    strides=1,
                                    padding="same")
        self.b3_conv3 = BasicConv2D(filters=256,
                                    kernel_size=(3, 1),
                                    strides=1,
                                    padding="same")
        self.b4_conv1 = BasicConv2D(filters=384,
                                    kernel_size=(1, 1),
                                    strides=1,
                                    padding="same")
        self.b4_conv2 = BasicConv2D(filters=448,
                                    kernel_size=(1, 3),
                                    strides=1,
                                    padding="same")
        self.b4_conv3 = BasicConv2D(filters=512,
                                    kernel_size=(3, 1),
                                    strides=1,
                                    padding="same")
        self.b4_conv4 = BasicConv2D(filters=256,
                                    kernel_size=(3, 1),
                                    strides=1,
                                    padding="same")
        self.b4_conv5 = BasicConv2D(filters=256,
                                    kernel_size=(1, 3),
                                    strides=1,
                                    padding="same")

    def call(self, inputs, training=None, **kwargs):
        b1 = self.b1_pool(inputs)
        b1 = self.b1_conv(b1, training=training)

        b2 = self.b2_conv(inputs, training=training)

        b3 = self.b3_conv1(inputs, training=training)
        b3_1 = self.b3_conv2(b3, training=training)
        b3_2 = self.b3_conv3(b3, training=training)

        b4 = self.b4_conv1(inputs, training=training)
        b4 = self.b4_conv2(b4, training=training)
        b4 = self.b4_conv3(b4, training=training)
        b4_1 = self.b4_conv4(b4, training=training)
        b4_2 = self.b4_conv5(b4, training=training)

        return tf.concat(values=[b1, b2, b3_1, b3_2, b4_1, b4_2], axis=-1)
