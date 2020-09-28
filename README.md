# retinanet

## Introduction
Implementation of retinanet with resnet backbones, which is compatible with tensorflow 2.2.0 and newer versions. `tf.data` is used to parse `tfrecords` and feed data to the model, which promises data parallelism. The implementation is heavily inspired by [fizyr](https://github.com/fizyr/keras-retinanet)


## Data Format
currently only pascal-voc xml format is supported

## Training on custom data

### create tfrecords from pascal-voc format
Use `retinanet_prepare` cli command to create tfrecords from a dataset of pascal-voc xml format. This will create a folder called `DATA` in the present working directory and place the tfrecords inside it.

### train
- Use `retinanet_train` cli command to start training.
- Use `freeze` function to freeze a trained model from [train_script.py](https://github.com/pk00095/retinanet/blob/master/retinanet/train_script.py)

### predict
- Load model using `load_model` function from [predict_script.py](https://github.com/pk00095/retinanet/blob/master/predict_script.py)
- Predict on a list of images using `predict` function from [predict_script.py](https://github.com/pk00095/retinanet/blob/master/predict_script.py)
