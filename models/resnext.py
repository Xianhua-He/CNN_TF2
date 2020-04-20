import tensorflow as tf
from models.resnext_block import build_ResNeXt_block
from configuration import NUM_CLASSES
from tensorflow.keras import layers

class ResNeXt(tf.keras.Model):
    def __init__(self, repeat_num_list, cardinality):
        if len(repeat_num_list) != 4:
            raise ValueError("The length of repeat_num_list, must be four")
        super(ResNeXt, self).__init__()
        self.conv1 = layers.Conv2D(64, (7, 7), 2, "same")
        self.bn1 = layers.BatchNormalization()
        self.pool1 = layers.MaxPool2D(3, 2, "same")
        self.block1 = build_ResNeXt_block(128, 1, cardinality, repeat_num_list[0])
        self.block2 = build_ResNeXt_block(filters=256,
                                          strides=2,
                                          groups=cardinality,
                                          repeat_num=repeat_num_list[1])
        self.block3 = build_ResNeXt_block(filters=512,
                                          strides=2,
                                          groups=cardinality,
                                          repeat_num=repeat_num_list[2])
        self.block4 = build_ResNeXt_block(filters=1024,
                                          strides=2,
                                          groups=cardinality,
                                          repeat_num=repeat_num_list[3])
        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(NUM_CLASSES, activation=tf.keras.activations.softmax)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)

        x = self.block1(x, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        x = self.block4(x, training=training)

        x = self.pool2(x)
        x = self.fc(x)

        return x


def ResNeXt50():
    return ResNeXt(repeat_num_list=[3, 4, 6, 3], cardinality=32)

def ResNeXt101():
    return ResNeXt(repeat_num_list=[3, 4, 23, 3],
                   cardinality=32)

