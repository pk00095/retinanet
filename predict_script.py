from tensorflow import keras
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess_input
from retinanet import get_retinanet_r50, retinanet_bbox

import os

import preprocessing

AnchorParameters_default = preprocessing.AnchorParameters.default

def load_model():
    num_classes = 4
    checkpoint_path = 'checkpoints/retinanet__pred.h5'

    sizes = AnchorParameters_default.sizes 
    ratios = AnchorParameters_default.ratios 
    scales = AnchorParameters_default.scales 
    strides = AnchorParameters_default.strides

    feature_extractor = get_retinanet_r50(
        num_classes,
        num_anchors_per_location=len(scales)*len(ratios),
        weights=None)

    #convert training_model to prediction model

    pred_model = retinanet_bbox(
        model=feature_extractor,
        sizes=sizes,
        strides=strides,
        ratios=ratios,
        scales=scales)

    pred_model.load_weights(
    	filepath=checkpoint_path,
    	by_name=True)

    print('Weights are Loaded ...')

if __name__ == '__main__':
	load_model()
