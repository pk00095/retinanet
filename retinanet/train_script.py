"""Main script which trains algorithm, and converts trained model to prediction model
"""
from tensorflow import keras
from tensorflow.keras.applications.resnet import preprocess_input as resnet_normalize_image
import os

from .retinanet_util import _BACKBONES, retinanet_bbox
from .losses import focal, smooth_l1
from .tfrecord_parser import parse_tfrecords
from .preprocessing import AnchorParameters

from segmind_track import KerasCallback
from segmind_track import set_experiment
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", help="the directory containing images", type=str, required=True)
    parser.add_argument("--ex_id", help="the directory containing images", type=str, required=True)
    parser.add_argument("--epochs", help="number of epochs to run training", type=int, required=True)
    parser.add_argument("--steps_per_epoch", help="the number of steps for a complete epoch", type=int, required=True)
    parser.add_argument("--snapshot_epoch", help="take snapshot every nth epoch", type=int, default=5)
    parser.add_argument("--batch_size", help="number of training instances per batch", type=int, default=2)
    parser.add_argument("--min_side", help="number of training instances per batch", type=int, default=800)
    parser.add_argument("--max_side", help="number of training instances per batch", type=int, default=1333)

    args = parser.parse_args()

    set_experiment(args.ex_id)

    trainer(
        num_classes=args.num_classes,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        snapshot_epoch=args.snapshot_epoch,
        batch_size=args.batch_size,
        min_side=args.min_side,
        max_side=args.max_side)


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
    prefix='retinanet__{epoch:02d}',
    callbacks=[KerasCallback()]):
    """This function builds the model and starts the training loop
    
    Args:
        num_classes (int): number of classes in training set
        epochs (int): number of epochs to train for
        steps_per_epoch (int): number of steps to run to finish an epoch
        snapshot_epoch (int): if `snapshot_epoch=2`, then a snapshot is taken at every 2nd consecutive epoch
        training_tfrecords (str, optional): glob pattern to tfrecords, defaults to `./DATA/train*.tfrecord`
        backbone (str, optional): One of 'resnet50/101/152'
        anchor_params (TYPE, optional): An instance of retinanet.AnchorParameters
        min_side (int, optional): minimum dimension, defaults to 800
        max_side (int, optional): maximum dimension, defaults to 1333
        optimizer (TYPE, optional): Optimizer instance which will be passed to model.compile()
        batch_size (int, optional): Batch-size
        tensorboard_dir (str, optional): Description
        checkpoint_path (str, optional): Description
        prefix (str, optional): Description
        callbacks (list, optional): Description
    """
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
    """This function converts a training model to prediction model
    
    Args:
        checkpoint_path (str): Path to where the checkpoint, which will be converted
        model_savepath (str): Path to where will be the prediction model be stored
        anchor_params (TYPE, optional): The instance of retinanet.AnchorParameters, which was used while training
        allow_class_overlap (bool, optional): if true then nms will be applied classwise
        apply_nms (bool, optional): Description
        nms_threshold (float, optional): Description
        score_threshold (float, optional): Description
        max_bboxes (int, optional): Description
    """
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