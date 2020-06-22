from tensorflow import keras
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess_input
from retinanet import get_retinanet_r50
from losses import focal, smooth_l1
from tfrecord_parser import parse_tfrecords
import os


def main():
    num_classes = 4
    learning_rate = 1e-4


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


    train_dataset_function = parse_tfrecords(
          filenames=os.path.join(os.getcwd(),'DATA','train*.tfrecord'), 
          batch_size=2, 
          num_classes=num_classes,
          preprocess_fn=resnet_preprocess_input)
          # sizes=AnchorParameters_default.sizes, 
          # ratios=AnchorParameters_default.ratios, 
          # scales=AnchorParameters_default.scales, 
          # strides=AnchorParameters_default.strides, 
          # min_side=800, 
          # max_side=1333)

    model = get_retinanet_r50(num_classes)
    model.compile(
        loss={
            'regression'    : smooth_l1(),
            'classification': focal(),
        },
        optimizer=keras.optimizers.Adam(lr=learning_rate, clipnorm=0.001))

    # print(model.summary())
    # exit()

    model.fit(
        train_dataset_function, 
        epochs=5, 
        steps_per_epoch=154,
        callbacks=[reduce_lr_callback])


if __name__ == '__main__':
    main()