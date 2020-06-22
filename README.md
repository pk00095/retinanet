# retinanet

## Introduction
create retinanet from tf.keras.applications, which is compatible with tensorflow 2.2.0 and newer versions and use tf.data api to feed data to algorithm. The implementation is heavily inspired by [fizyr](https://github.com/fizyr/keras-retinanet)


## Data Format
currently only pascal-voc xml format is supported

## Training on custom data

### create tfrecords from pascal-voc format
Edit [tfrecord_creator.py](https://github.com/pk00095/retinanet/blob/master/tfrecord_creator.py)

### train
- Use `trainer` function to train on your data [train_script.py](https://github.com/pk00095/retinanet/blob/master/train_script.py)
- Use `freeze` function to freeze a trained model from [train_script.py](https://github.com/pk00095/retinanet/blob/master/train_script.py)

### predict
- Load model using `load_model` function from [predict_script.py](https://github.com/pk00095/retinanet/blob/master/predict_script.py)
- Predict on a list of images using `predict` function from [predict_script.py](https://github.com/pk00095/retinanet/blob/master/predict_script.py)
