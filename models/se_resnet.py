import tensorflow as tf
from configuration import NUM_CLASSES
from tensorflow.keras import layers


class SEBlock(tf.keras.layers.Layer):
    def __init__(self, input_channels, r=16):
        super(SEBlock, self).__init__()
        self.pool = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(input_channels//r)
        self.fc2 = layers.Dense(input_channels)

    def call(self, inputs, **kwargs):
        branch = self.pool(inputs)
        branch = self.fc1(branch)
        branch = tf.nn.relu(branch)
        branch = self.fc2(branch)
        branch = tf.nn.sigmoid(branch)
        branch = tf.expand_dims(input=branch, axis=1)
        branch = tf.expand_dims(input=branch, axis=1)
        output = layers.multiply(inputs=[inputs, branch])
        return output


class BottleNeck(tf.keras.layers.Layer):
    def __init__(self, filter_num, strides=1):
        super(BottleNeck, self).__init__()
        self.conv1 = layers.Conv2D(filter_num, 1, 1, "same")
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filter_num, 3, strides, "same")
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(filter_num*4, 1, 1, "same")
        self.bn3 = layers.BatchNormalization()
        self.se = SEBlock(input_channels=filter_num*4)
        self.downsample = tf.keras.Sequential()
        self.downsample.add(layers.Conv2D(filter_num*4, 1, strides))
        self.downsample.add(layers.BatchNormalization())

    def call(self, inputs, training=None):
        identity = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.se(x)
        output = tf.nn.relu(layers.add([identity, x]))
        return output


class SEResNet(tf.keras.Model):
    def __init__(self, block_num):
        super(SEResNet, self).__init__()
        self.pre1 = layers.Conv2D(64, 7, 2, "same")
        self.pre2 = layers.BatchNormalization()
        self.pre3 = layers.Activation(tf.keras.activations.relu)
        self.pre4 = layers.MaxPool2D(3, 2)
        self.layer1 = self._make_res_block(64, block_num[0])
        self.layer2 = self._make_res_block(128, block_num[1], 2)
        self.layer3 = self._make_res_block(filter_num=256,
                                           blocks=block_num[2],
                                           stride=2)
        self.layer4 = self._make_res_block(filter_num=512,
                                           blocks=block_num[3],
                                           stride=2)

    def _make_res_block(self, filter_num, blocks, strides=1):
        res_block = tf.keras.Sequential()
        res_block.add(BottleNeck(filter_num, strides))
        for _ in range(1, blocks):
            res_block.add(BottleNeck(filter_num, 1))
        return res_block

    def call(self, inputs, training=None, mask=None):
        pre1 = self.pre1(inputs)
        pre2 = self.pre2(pre1, training=training)
        pre3 = self.pre3(pre2)
        pre4 = self.pre4(pre3)
        l1 = self.layer1(pre4, training=training)
        l2 = self.layer2(l1, training=training)
        l3 = self.layer3(l2, training=training)
        l4 = self.layer4(l3, training=training)
        avgpool = self.avgpool(l4)
        out = self.fc(avgpool)
        return out

def se_resnet_50():
    return SEResNet(block_num=[3, 4, 6, 3])


def se_resnet_101():
    return SEResNet(block_num=[3, 4, 23, 3])


def se_resnet_152():
    return SEResNet(block_num=[3, 8, 36, 3])

