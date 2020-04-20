# Basic_CNN_Tensoflow2

# Networks included
+ ResNet
+ MobileNetV1
+ MobileNetV2
+ ShuffleNetV2
+ InceptionV4
+ ResNeXt
+ DenseNet


## train
1. Requirements:
+ Python >= 3.6
+ Tensorflow >= 2.0.0
2. To train the network on your own dataset, you can put the dataset under the folder **original dataset**, and the directory should look like this:
```
|——original dataset
   |——class_name_0
   |——class_name_1
   |——class_name_2
   |——class_name_3
```
3. Run the script **split_datast.py** to split the raw dataset
```
|——dataset
   |——train
        |——class_name_1
        |——class_name_2
        ......
        |——class_name_n
   |——valid
        |——class_name_1
        |——class_name_2
        ......
        |——class_name_n
   |—-test
        |——class_name_1
        |——class_name_2
        ......
        |——class_name_n
```
4. Run **to_tfrecord.py** to generate tfrecord files.
5. Change the corresponding parameters in **config.py**.
6. Run **train.py** to start training.<br/>

## Evaluate
Run **evaluate.py** to evaluate the model's performance on the test dataset.