# retinanet

## Introduction
create retinanet from tf.keras.applications, which is compatible with tensorflow 2.2.0 and newer versions and use tf.data api to feed data to algorithm. The implementation is heavily inspired by [fizyr](https://github.com/fizyr/keras-retinanet)


## Data Format
currently only pascal-voc xml format is supported

## Training on custom data
### create tfrecords from pascal-voc format
Edit [tfrecord_creator.py](https://github.com/pk00095/retinanet/blob/master/tfrecord_creator.py)
### train
Edit [train_script.py](https://github.com/pk00095/retinanet/blob/master/train_script.py)
### predict
Edit [predict_script.py](https://github.com/pk00095/retinanet/blob/master/predict_script.py
