from tensorflow import keras
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess_input
from retinanet import get_retinanet_r50, retinanet_bbox
from losses import focal, smooth_l1
from tfrecord_parser import parse_tfrecords
import os

import preprocessing

AnchorParameters_default = preprocessing.AnchorParameters.default

def trainer():
    num_classes = 4
    learning_rate = 1e-4
    batch_size = 2
    tensorboard_dir = 'logs'

    checkpoint_path = 'checkpoints'

    prefix = 'retinanet__{epoch:02d}.h5'

    sizes = AnchorParameters_default.sizes 
    ratios = AnchorParameters_default.ratios 
    scales = AnchorParameters_default.scales 
    strides = AnchorParameters_default.strides

    os.makedirs(checkpoint_path, exist_ok=True)

    reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(
        monitor    = 'loss',
        factor     = 0.1,
        patience   = 2,
        verbose    = 1,
        mode       = 'auto',
        min_delta  = 0.0001,
        cooldown   = 0,
        min_lr     = 0
    )

    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=tensorboard_dir, 
        histogram_freq=0, 
        write_graph=True, 
        write_images=False,
        update_freq=2, 
        profile_batch=2, 
        embeddings_freq=0,
        embeddings_metadata=None)

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        os.path.join(checkpoint_path, prefix), 
        monitor='val_loss', 
        verbose=0, 
        save_best_only=True,
        save_weights_only=False, 
        mode='auto', 
        save_freq=2)


    train_dataset_function = parse_tfrecords(
          filenames=os.path.join(os.getcwd(),'DATA','train*.tfrecord'), 
          batch_size=batch_size, 
          num_classes=num_classes,
          preprocess_fn=resnet_preprocess_input,
          sizes=sizes, 
          ratios=ratios, 
          scales=scales, 
          strides=strides,) 
          # min_side=800, 
          # max_side=1333)

    model = get_retinanet_r50(
        num_classes,
        num_anchors_per_location=AnchorParameters_default.num_anchors())

    model.compile(
        loss={
            'regression'    : smooth_l1(),
            'classification': focal()},
        optimizer=keras.optimizers.Adam(lr=learning_rate, clipnorm=0.001))

    print(model.summary())

    model.fit(
        train_dataset_function, 
        epochs=5, 
        steps_per_epoch=154,
        callbacks=[reduce_lr_callback, model_checkpoint_callback])

    print('saving model')
    model.save('./checkpoints/model')

def freeze(
    checkpoint_path,
    model_savepath,
    sizes = AnchorParameters_default.sizes,
    ratios = AnchorParameters_default.ratios,
    scales = AnchorParameters_default.scales,
    strides = AnchorParameters_default.strides,
    allow_class_overlap = False,
    apply_nms= True,
    nms_threshold = 0.5,
    score_threshold = 0.05,
    max_bboxes = 300
    ):

    # feature_extractor = get_retinanet_r50(
    #     num_classes,
    #     num_anchors_per_location=len(scales)*len(ratios),
    #     weights=None)

    # feature_extractor.load_weights(
    #     filepath=checkpoint_path,
    #     by_name=True)

    print('Loading weights from trained file ..')
    # compile==False to avoid ValueError: Unknown loss function:_smooth_l1
    feature_extractor = keras.models.load_model(filepath=checkpoint_path, compile=False) 

    #convert training_model to prediction model
    print('converting training model to prediction model ...')

    pred_model = retinanet_bbox(
        model=feature_extractor,
        sizes=sizes,
        strides=strides,
        ratios=ratios,
        scales=scales,
        nms                   = apply_nms,
        class_specific_filter = not allow_class_overlap,
        nms_threshold         = nms_threshold,
        score_threshold       = score_threshold,
        max_detections        = max_bboxes)

    pred_model.save(filepath=model_savepath)

    print('Done saving ...')


if __name__ == '__main__':
    trainer()
    freeze('./checkpoints/model', './checkpoints/prediction')