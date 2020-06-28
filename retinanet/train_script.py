from tensorflow import keras
from tensorflow.keras.applications.resnet import preprocess_input as resnet_normalize_image
import os

from .retinanet_util import _BACKBONES, retinanet_bbox
from .losses import focal, smooth_l1
from .tfrecord_parser import parse_tfrecords
from .preprocessing import AnchorParameters

def trainer(
    num_classes,
    epochs,
    steps_per_epoch,
    snapshot_epoch,
    training_tfrecords=os.path.join(os.getcwd(),'DATA','train*.tfrecord'),
    backbone='resnet50',
    anchor_params=AnchorParameters(),
    min_side=800,
    max_side=1333,
    optimizer=keras.optimizers.Adam(lr=1e-4, clipnorm=0.001),
    batch_size=2,
    tensorboard_dir='logs',
    checkpoint_path='checkpoints',
    prefix='retinanet__{epoch:02d}.h5',
    callbacks=[]):


    assert backbone in _BACKBONES, 'Only {} are supported as backbones'.format(_BACKBONES.keys())
    assert isinstance(anchor_params, AnchorParameters), 'please pass a retinanet.preprocessing.AnchorParameters object to arg `anchor_params`'

    sizes = anchor_params.sizes 
    ratios = anchor_params.ratios 
    scales = anchor_params.scales 
    strides = anchor_params.strides
    num_anchors_per_location = anchor_params.num_anchors()

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

    callbacks.append(reduce_lr_callback)

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        os.path.join(checkpoint_path, prefix), 
        # monitor='val_loss', 
        verbose=1, 
        save_best_only=True,
        save_weights_only=False, 
        mode='auto', 
        period=snapshot_epoch)

    callbacks.append(model_checkpoint_callback)

    train_dataset_function = parse_tfrecords(
          filenames=training_tfrecords, 
          batch_size=batch_size, 
          num_classes=num_classes,
          sizes=sizes, 
          ratios=ratios, 
          scales=scales, 
          strides=strides, 
          min_side=min_side, 
          max_side=max_side,
          preprocess_fn=resnet_normalize_image)

    model = _BACKBONES[backbone](
        num_classes,
        num_anchors_per_location=num_anchors_per_location)

    model.compile(
        loss={
            'regression'    : smooth_l1(),
            'classification': focal()},
        optimizer=optimizer)

    print(model.summary())

    model.fit(
        train_dataset_function, 
        epochs=epochs, 
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks)

    print('saving model')
    model.save(os.path.join(checkpoint_path,'model'))

def freeze(
    checkpoint_path,
    model_savepath,
    anchor_params=AnchorParameters(),
    allow_class_overlap=False,
    apply_nms=True,
    nms_threshold=0.5,
    score_threshold=0.05,
    max_bboxes=300
    ):

    assert isinstance(anchor_params, AnchorParameters), 'please pass a retinanet.preprocessing.AnchorParameters object to arg `anchor_params`'

    sizes = anchor_params.sizes 
    ratios = anchor_params.ratios 
    scales = anchor_params.scales 
    strides = anchor_params.strides
    num_anchors_per_location = anchor_params.num_anchors()

    print('Loading weights from trained file ..')
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