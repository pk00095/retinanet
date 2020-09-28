import argparse
from retinanet import tfrecord_creator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", help="the directory containing images", type=str, required=True)
    parser.add_argument("--xml_dir", help="the directory containing annotations in pascal-voc xml format", type=str, required=True)

    args = parser.parse_args()

    tfrecord_creator.create_tfrecords(
        image_dir=args.image_dir, 
        xml_dir=args.xml_dir)
