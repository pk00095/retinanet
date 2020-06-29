import unittest, zipfile, tempfile, os
import tensorflow as tf
import numpy as np

class Test_RetinaNet(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        zip_url = 'https://segmind-data.s3.ap-south-1.amazonaws.com/edge/data/aerial-vehicles-dataset.zip'
        path_to_zip_file = tf.keras.utils.get_file(
            'aerial-vehicles-dataset.zip',
            zip_url,
            cache_dir=tempfile.gettempdir(), 
            cache_subdir='',
            extract=False)
        directory_to_extract_to = os.path.join(tempfile.gettempdir(),'aerial-vehicles-dataset')
        with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
            zip_ref.extractall(directory_to_extract_to)

        cls.dataset = directory_to_extract_to

    def setup(self):
        self.dataset = cls.dataset

    def test_build_tfrecord(self):
        from retinanet.tfrecord_creator import create_tfrecords

        create_tfrecords(
            image_dir=os.path.join(self.dataset, 'images'),
            xml_dir=os.path.join(self.dataset, 'annotations','pascalvoc_xml'),
            outpath=os.path.join(tempfile.gettempdir(),'DATA'),
            split_name='train')

    def test_data_augmentation(self):
        from retinanet.tfrecord_parser import parse_tfrecords
        from retinanet.preprocessing import AnchorParameters

        from albumentations import (
            RandomContrast, RandomCrop, Rotate, RandomGamma, Flip, OneOf, MotionBlur, MedianBlur, Blur, 
            RGBShift, RandomBrightness, RandomBrightnessContrast, RandomContrast, 
            RandomFog, RandomGamma, RandomRain, RandomShadow, RandomSnow, RandomSunFlare,  
            RandomRotate90, CenterCrop)

        aug = [OneOf(
                    [ #RandomFog(fog_coef_lower=1,fog_coef_upper=1,alpha_coef=0.05,p=1.0),
                    # RandomBrightnessContrast(brightness_limit=0.2,contrast_limit=10.5,p=1.0),
                    # RandomShadow(shadow_roi=(0,0.5,1,1),num_shadows_lower=1,num_shadows_upper=2,shadow_dimension=5,p=0.5),
                    RandomSnow(),
                    RandomSunFlare(),
                    Flip()],
                    p=0.8)]

        anchor_params = AnchorParameters()

        dataset = parse_tfrecords(
            filenames=os.path.join(tempfile.gettempdir(),'DATA','train*.tfrecord'), 
            batch_size=2,
            num_classes=5,
            sizes=anchor_params.sizes, 
            ratios=anchor_params.ratios, 
            scales=anchor_params.scales, 
            strides=anchor_params.strides,
            aug=aug)

        for data, annotation in dataset.take(10):
            image_batch = data.numpy()

            abxs_batch = annotation['regression'].numpy()
            labels_batch = annotation['classification'].numpy()

            # print(image_batch.shape, abxs_batch.shape, labels_batch.shape)
            # print(image_batch.dtype, abxs_batch.dtype, labels_batch.dtype)
            print((image_batch).max(), image_batch.min())

    def test_trainloop(self):
        from retinanet.train_script import trainer

        # trainer(
        # num_classes=4,
        # epochs=5,
        # steps_per_epoch=154,
        # snapshot_epoch=5,
        # training_tfrecords=os.path.join(os.getcwd(),'DATA','train*.tfrecord'))




if __name__ == '__main__':
    unittest.main()
