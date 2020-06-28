import unittest, zipfile, tempfile, os
import tensorflow as tf

class Test_Classification(unittest.TestCase):

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
            os.path.join(self.dataset, 'images'),
            os.path.join(self.dataset, 'annotations','pascalvoc_xml'),
            'aerial-vehicles-dataset.tfrecord')

    def test_trainloop(self):
        from retinanet.train_script import trainer

        trainer(
        num_classes=4,
        epochs=5,
        steps_per_epoch=154,
        snapshot_epoch=5)




if __name__ == '__main__':
    unittest.main()
