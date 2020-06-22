import tensorflow as tf
from tensorflow import keras
import numpy as np

import preprocessing
from postprocessing import RegressBoxes, ClipBoxes, FilterDetections, Anchors

# AnchorParameters_default = preprocessing.AnchorParameters.default


class PriorProbability(keras.initializers.Initializer):
    """Apply a prior probability to the weights.
    
    Attributes:
        probability (TYPE): Description
    """

    def __init__(self, probability=0.01):
        """Summary
        
        Args:
            probability (float, optional): Description
        """
        super(PriorProbability, self).__init__()
        self.probability = probability

    def get_config(self):
        """Summary
        
        Returns:
            TYPE: Description
        """
        return {
            'probability': self.probability
        }

    def __call__(self, shape, partition_info=None, dtype=None):
        """Summary
        
        Args:
            shape (TYPE): Description
            partition_info (None, optional): Description
            dtype (None, optional): Description
        
        Returns:
            TYPE: Description
        """
        # set bias to -log((1 - p)/p) for foreground
        result = tf.ones(shape, dtype=dtype) * -tf.math.log((1 - self.probability) / self.probability)
        return result


def resize_images(images, size, method='bilinear', align_corners=False):
    """See https://www.tensorflow.org/versions/r1.14/api_docs/python/tf/image/resize_images .
    Args
        method: The method used for interpolation. One of ('bilinear', 'nearest', 'bicubic', 'area').
    
    Args:
        images (TYPE): Description
        size (TYPE): Description
        method (str, optional): Description
        align_corners (bool, optional): Description
    
    Returns:
        TYPE: Description
    """
    return tf.compat.v1.image.resize_images(images, size, method, align_corners)



class UpsampleLike(keras.layers.Layer):
    """Keras layer for upsampling a Tensor to be the same shape as another Tensor.
    """

    def call(self, inputs, **kwargs):
        """Summary
        
        Args:
            inputs (TYPE): Description
            **kwargs: Description
        
        Returns:
            TYPE: Description
        """
        source, target = inputs
        #print(target)
        target_shape = keras.backend.shape(target)
        #print(target_shape)
        if keras.backend.image_data_format() == 'channels_first':
            source = backend.transpose(source, (0, 2, 3, 1))
            output = resize_images(source, (target_shape[2], target_shape[3]), method='nearest')
            output = backend.transpose(output, (0, 3, 1, 2))
            return output
        else:
            return resize_images(source, (target_shape[1], target_shape[2]), method='nearest')

    def compute_output_shape(self, input_shape):
        """Summary
        
        Args:
            input_shape (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        if keras.backend.image_data_format() == 'channels_first':
            return (input_shape[0][0], input_shape[0][1]) + input_shape[1][2:4]
        else:
            return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1],)



def default_classification_model(
    num_classes,
    num_anchors,
    pyramid_feature_size=256,
    prior_probability=0.01,
    classification_feature_size=256,
    name='classification_submodel'
):
    """Creates the default classification submodel.
    
    Args
        num_classes                 : Number of classes to predict a score for at each feature level.
        num_anchors                 : Number of anchors to predict classification scores for at each feature level.
        pyramid_feature_size        : The number of filters to expect from the feature pyramid levels.
        classification_feature_size : The number of filters to use in the layers in the classification submodel.
        name                        : The name of the submodel.
    
    Returns
        A keras.models.Model that predicts classes for each anchor.
    
    Args:
        num_classes (TYPE): Description
        num_anchors (TYPE): Description
        pyramid_feature_size (int, optional): Description
        prior_probability (float, optional): Description
        classification_feature_size (int, optional): Description
        name (str, optional): Description
    """
    options = {
        'kernel_size' : 3,
        'strides'     : 1,
        'padding'     : 'same',
    }

    # if keras.backend.image_data_format() == 'channels_first':
    #     inputs  = keras.layers.Input(shape=(pyramid_feature_size, None, None))
    # else:
    inputs  = keras.layers.Input(shape=(None, None, pyramid_feature_size))
    outputs = inputs

    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=classification_feature_size,
            activation='relu',
            name='pyramid_classification_{}'.format(i),
            kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None),
            bias_initializer='zeros',
            **options
        )(outputs)

    outputs = keras.layers.Conv2D(
        filters=num_classes * num_anchors,
        kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None),
        bias_initializer=PriorProbability(probability=prior_probability),
        name='pyramid_classification',
        **options
    )(outputs)

    # reshape output and apply sigmoid
    # if keras.backend.image_data_format() == 'channels_first':
    #     outputs = keras.layers.Permute((2, 3, 1), name='pyramid_classification_permute')(outputs)
    outputs = keras.layers.Reshape((-1, num_classes), name='pyramid_classification_reshape')(outputs)
    outputs = keras.layers.Activation('sigmoid', name='pyramid_classification_sigmoid')(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def default_regression_model(num_values, num_anchors, pyramid_feature_size=256, regression_feature_size=256, name='regression_submodel'):
    """Creates the default regression submodel.
    
    Args
        num_values              : Number of values to regress.
        num_anchors             : Number of anchors to regress for each feature level.
        pyramid_feature_size    : The number of filters to expect from the feature pyramid levels.
        regression_feature_size : The number of filters to use in the layers in the regression submodel.
        name                    : The name of the submodel.
    
    Returns
        A keras.models.Model that predicts regression values for each anchor.
    
    Args:
        num_values (TYPE): Description
        num_anchors (TYPE): Description
        pyramid_feature_size (int, optional): Description
        regression_feature_size (int, optional): Description
        name (str, optional): Description
    
    Returns:
        TYPE: Description
    """
    # All new conv layers except the final one in the
    # RetinaNet (classification) subnets are initialized
    # with bias b = 0 and a Gaussian weight fill with stddev = 0.01.
    options = {
        'kernel_size'        : 3,
        'strides'            : 1,
        'padding'            : 'same',
        'kernel_initializer' : tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer'   : 'zeros'
    }

    # if keras.backend.image_data_format() == 'channels_first':
    #     inputs  = keras.layers.Input(shape=(pyramid_feature_size, None, None))
    # else:
    inputs  = keras.layers.Input(shape=(None, None, pyramid_feature_size))
    outputs = inputs
    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=regression_feature_size,
            activation='relu',
            name='pyramid_regression_{}'.format(i),
            **options
        )(outputs)

    outputs = keras.layers.Conv2D(num_anchors * num_values, name='pyramid_regression', **options)(outputs)
    if keras.backend.image_data_format() == 'channels_first':
        outputs = keras.layers.Permute((2, 3, 1), name='pyramid_regression_permute')(outputs)
    outputs = keras.layers.Reshape((-1, num_values), name='pyramid_regression_reshape')(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def __create_pyramid_features(C3, C4, C5, feature_size=256):
    """Creates the FPN layers on top of the backbone features.
    
    Args
        C3           : Feature stage C3 from the backbone.
        C4           : Feature stage C4 from the backbone.
        C5           : Feature stage C5 from the backbone.
        feature_size : The feature size to use for the resulting feature levels.
    
    Returns
        A list of feature levels [P3, P4, P5, P6, P7].
    
    Args:
        C3 (TYPE): Description
        C4 (TYPE): Description
        C5 (TYPE): Description
        feature_size (int, optional): Description
    
    Returns:
        TYPE: Description
    """
    # upsample C5 to get P5 from the FPN paper
    P5           = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C5_reduced')(C5)
    P5_upsampled = UpsampleLike(name='P5_upsampled')([P5, C4])
    P5           = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P5')(P5)

    # add P5 elementwise to C4
    P4           = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced')(C4)
    P4           = keras.layers.Add(name='P4_merged')([P5_upsampled, P4])
    P4_upsampled = UpsampleLike(name='P4_upsampled')([P4, C3])
    P4           = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4')(P4)

    # add P4 elementwise to C3
    P3 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced')(C3)
    P3 = keras.layers.Add(name='P3_merged')([P4_upsampled, P3])
    P3 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3')(P3)

    # "P6 is obtained via a 3x3 stride-2 conv on C5"
    P6 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P6')(C5)

    # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
    P7 = keras.layers.Activation('relu', name='C6_relu')(P6)
    P7 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P7')(P7)

    return [P3, P4, P5, P6, P7]


def default_submodels(num_classes, num_anchors):
    """Create a list of default submodels used for object detection.
    
    The default submodels contains a regression submodel and a classification submodel.
    
    Args
        num_classes : Number of classes to use.
        num_anchors : Number of base anchors.
    
    Returns
        A list of tuple, where the first element is the name of the submodel and the second element is the submodel itself.
    
    Args:
        num_classes (TYPE): Description
        num_anchors (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    return [
        ('regression', default_regression_model(4, num_anchors)),
        ('classification', default_classification_model(num_classes, num_anchors))
    ]


def __build_model_pyramid(name, model, features):
    """Applies a single submodel to each FPN level.
    
    Args
        name     : Name of the submodel.
        model    : The submodel to evaluate.
        features : The FPN features.
    
    Returns
        A tensor containing the response from the submodel on the FPN features.
    
    Args:
        name (TYPE): Description
        model (TYPE): Description
        features (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    return keras.layers.Concatenate(axis=1, name=name)([model(f) for f in features])


def __build_pyramid(models, features):
    """Applies all submodels to each FPN level.
    
    Args
        models   : List of submodels to run on each pyramid level (by default only regression, classifcation).
        features : The FPN features.
    
    Returns
        A list of tensors, one for each submodel.
    
    Args:
        models (TYPE): Description
        features (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    return [__build_model_pyramid(n, m, features) for n, m in models]


def __build_anchors(features, sizes, strides, ratios, scales):
    """Builds anchors for the shape of the features from FPN.
    
    Args
        anchor_parameters : Parameteres that determine how anchors are generated.
        features          : The FPN features.
    
    Returns
        A tensor containing the anchors for the FPN features.
    
        The shape is:
        ```
        (batch_size, num_anchors, 4)
        ```
    
    Args:
        features (TYPE): Description
        sizes (TYPE, optional): Description
        strides (TYPE, optional): Description
        ratios (TYPE, optional): Description
        scales (TYPE, optional): Description
    
    Returns:
        TYPE: Description
    """
    anchors = [
        Anchors(
            size=sizes[i],
            stride=strides[i],
            ratios=ratios,
            scales=scales,
            name='anchors_{}'.format(i)
        )(f) for i, f in enumerate(features)
    ]

    return keras.layers.Concatenate(axis=1, name='anchors')(anchors)


def retinanet(
    inputs,
    backbone_layers,
    num_classes,
    num_anchors,
    create_pyramid_features = __create_pyramid_features,
    submodels               = None,
    name                    = 'retinanet'
):
    """Construct a RetinaNet model on top of a backbone.
    
    This model is the minimum model necessary for training (with the unfortunate exception of anchors as output).
    
    Args
        inputs                  : keras.layers.Input (or list of) for the input to the model.
        num_classes             : Number of classes to classify.
        num_anchors             : Number of base anchors.
        create_pyramid_features : Functor for creating pyramid features given the features C3, C4, C5 from the backbone.
        submodels               : Submodels to run on each feature map (default is regression and classification submodels).
        name                    : Name of the model.
    
    Returns
        A keras.models.Model which takes an image as input and outputs generated anchors and the result from each submodel on every pyramid level.
    
        The order of the outputs is as defined in submodels:
        ```
        [
            regression, classification, other[0], other[1], ...
        ]
        ```
    
    Args:
        inputs (TYPE): Description
        backbone_layers (TYPE): Description
        num_classes (TYPE): Description
        num_anchors (TYPE): Description
        create_pyramid_features (TYPE, optional): Description
        submodels (None, optional): Description
        name (str, optional): Description
    """

    if submodels is None:
        # Regression head(keras.Model with inp (None,None,256)) and classification head(keras.Model with inp (None,None,256))
        submodels = default_submodels(num_classes, num_anchors)

    C3, C4, C5 = backbone_layers

    # compute pyramid features as per https://arxiv.org/abs/1708.02002
    # i.e [P3, P4, P5, P6, P7] where each layer has 256 channels
    features = create_pyramid_features(C3, C4, C5)

    # for all pyramid levels, run available submodels
    pyramids = __build_pyramid(submodels, features)

    return keras.models.Model(inputs=inputs, outputs=pyramids, name=name)


def get_retinanet_r50(num_classes, num_anchors_per_location, weights='imagenet'):
    """Summary
    
    Args:
        num_classes (TYPE): Description
        num_anchors_per_location (TYPE, optional): Description
    
    Returns:
        TYPE: Description
    """
    inputs = keras.layers.Input(shape=(None, None, 3))

    resnet = keras.applications.resnet.ResNet50(include_top=False, weights=weights, input_tensor=inputs, pooling=None)

    C2 = resnet.get_layer('conv2_block3_out').output
    C3 = resnet.get_layer('conv3_block4_out').output
    C4 = resnet.get_layer('conv4_block6_out').output
    C5 = resnet.get_layer('conv5_block3_out').output

    #print(C3, C4, C5)

    return retinanet(
        inputs=inputs, 
        backbone_layers=(C3,C4,C5), 
        num_classes=num_classes,
        num_anchors=num_anchors_per_location)
    


def retinanet_bbox(
    model,
    sizes,
    strides,
    ratios,
    scales,
    nms                   = True,
    class_specific_filter = True,
    name                  = 'retinanet-bbox',
    anchor_params         = None,
    nms_threshold         = 0.5,
    score_threshold       = 0.05,
    max_detections        = 300,
    parallel_iterations   = 32,
    **kwargs
):
    """ Construct a RetinaNet model on top of a backbone and adds convenience functions to output boxes directly.

    This model uses the minimum retinanet model and appends a few layers to compute boxes within the graph.
    These layers include applying the regression values to the anchors and performing NMS.

    Args
        model                 : RetinaNet model to append bbox layers to. If None, it will create a RetinaNet model using **kwargs.
        nms                   : Whether to use non-maximum suppression for the filtering step.
        class_specific_filter : Whether to use class specific filtering or filter for the best scoring class only.
        name                  : Name of the model.
        anchor_params         : Struct containing anchor parameters. If None, default values are used.
        nms_threshold         : Threshold for the IoU value to determine when a box should be suppressed.
        score_threshold       : Threshold used to prefilter the boxes with.
        max_detections        : Maximum number of detections to keep.
        parallel_iterations   : Number of batch items to process in parallel.
        **kwargs              : Additional kwargs to pass to the minimal retinanet model.

    Returns
        A keras.models.Model which takes an image as input and outputs the detections on the image.

        The order is defined as follows:
        ```
        [
            boxes, scores, labels, other[0], other[1], ...
        ]
        ```
    """

    # if no anchor parameters are passed, use default values

    # compute the anchors
    features = [model.get_layer(p_name).output for p_name in ['P3', 'P4', 'P5', 'P6', 'P7']]

    anchors  = __build_anchors(
        features,    
        sizes=sizes,
        strides=strides,
        ratios=ratios,
        scales=scales)

    # we expect the anchors, regression and classification values as first output
    regression     = model.outputs[0]
    classification = model.outputs[1]

    # "other" can be any additional output from custom submodels, by default this will be []
    # other = model.outputs[2:]

    # apply predicted regression to anchors
    boxes = RegressBoxes(name='boxes')([anchors, regression])
    boxes = ClipBoxes(name='clipped_boxes')([model.inputs[0], boxes])

    # filter detections (apply NMS / score threshold / select top-k)
    detections = FilterDetections(
        nms                   = nms,
        class_specific_filter = class_specific_filter,
        name                  = 'filtered_detections',
        nms_threshold         = nms_threshold,
        score_threshold       = score_threshold,
        max_detections        = max_detections,
        parallel_iterations   = parallel_iterations
    )([boxes, classification])# + other)

    # construct the model
    return keras.models.Model(inputs=model.inputs, outputs=detections, name=name)

def main():
    """Summary
    """
    inputs = keras.layers.Input(shape=(None, None, 3))

    resnet = keras.applications.resnet.ResNet50(include_top=False, weights='imagenet', input_tensor=inputs, pooling=None)

    C2 = resnet.get_layer('conv2_block3_out').output
    C3 = resnet.get_layer('conv3_block4_out').output
    C4 = resnet.get_layer('conv4_block6_out').output
    C5 = resnet.get_layer('conv5_block3_out').output

    #print(C3, C4, C5)

    retinanet_model = retinanet(inputs=inputs, backbone_layers=(C3,C4,C5), num_classes=4)

    print(retinanet_model.summary())
    #print(retinanet_model.outputs)

if __name__ == '__main__':
    main()
