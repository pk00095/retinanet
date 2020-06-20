"""Summary
"""
import tensorflow as tf
import cv2, os, glob
import numpy as np
import pdb

def parse_tfrecords(filenames, batch_size, min_side=800, max_side=1333):
    """Summary
    
    Args:
        filenames (TYPE): Description
        batch_size (TYPE): Description
        min_side (int, optional): Description
        max_side (int, optional): Description
    
    Returns:
        tf.data.Dataset: Description
    """
    def pad_resize(image, height, width, scale):
        """Summary
        
        Args:
            image (TYPE): Description
            height (TYPE): Description
            width (TYPE): Description
            scale (TYPE): Description
        
        Returns:
            numpy nd.array: Description
        """
        padded_image = np.zeros(shape=(height.astype(int), width.astype(int),3), dtype=image.dtype)
        h,w,_ =  image.shape
        padded_image[:h,:w,:] = image
        return cv2.resize(padded_image, None, fx=scale, fy=scale)

    @tf.function
    def decode_pad_resize(image_string, pad_height, pad_width, scale):
      """Summary
      
      Args:
          image_string (TYPE): Description
          pad_height (TYPE): Description
          pad_width (TYPE): Description
          scale (TYPE): Description
      
      Returns:
          tf.tensor: Description
      """
      image = tf.image.decode_jpeg(image_string)
      image = tf.numpy_function(pad_resize, [image, pad_height, pad_width, scale], Tout=image.dtype)
      #image.set_shape([None, None, 3])
      return image

    def _parse_function(serialized):
        """Summary
        
        Args:
            serialized (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        features = {
          'image/height': tf.io.FixedLenFeature([], tf.int64),
          'image/width': tf.io.FixedLenFeature([], tf.int64),
          'image/encoded': tf.io.FixedLenFeature([],tf.string),
          'image/object/bbox/xmin': tf.io.VarLenFeature(tf.keras.backend.floatx()),
          'image/object/bbox/xmax': tf.io.VarLenFeature(tf.keras.backend.floatx()),
          'image/object/bbox/ymin': tf.io.VarLenFeature(tf.keras.backend.floatx()),
          'image/object/bbox/ymax': tf.io.VarLenFeature(tf.keras.backend.floatx()),
          'image/f_id': tf.io.FixedLenFeature([], tf.int64),
          'image/object/class/label':tf.io.VarLenFeature(tf.int64)}


        parsed_example = tf.io.parse_example(serialized=serialized, features=features)

        max_height = tf.cast(tf.keras.backend.max(parsed_example['image/height']), tf.int32)
        max_width = tf.cast(tf.keras.backend.max(parsed_example['image/width']), tf.int32)

        smallest_side = tf.keras.backend.minimum(max_height, max_width)
        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side
        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = tf.keras.backend.maximum(max_height, max_width)

        # scale = tf.cond(largest_side * tf.cast(scale, tf.int32) > max_side, lambda: max_side / largest_side, , lambda: scale)
        if largest_side * tf.cast(scale, tf.int32) > max_side:
            scale = max_side / largest_side

        scale = tf.cast(scale, tf.keras.backend.floatx()) 

        image_batch = tf.map_fn(lambda x: decode_pad_resize(x, max_height, max_width, scale), parsed_example['image/encoded'], dtype=tf.uint8)
        #print(scale)

        xmin_batch = parsed_example['image/object/bbox/xmin']*scale
        xmax_batch = parsed_example['image/object/bbox/xmax']*scale
        ymin_batch = parsed_example['image/object/bbox/ymin']*scale
        ymax_batch = parsed_example['image/object/bbox/ymax']*scale

        annotation_batch = []

        label_batch = parsed_example['image/object/class/label']

        print(xmin_batch.shape)
        print(ymin_batch.shape)
        print(xmax_batch.shape)
        print(ymax_batch.shape)
        print(label_batch.shape)


        # create GT from annotations
        # augment image_batch
        # normalize image through preprocessing


        return image_batch #, {'regression':abxs, 'classification':lbls}


    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.interleave(
      tf.data.TFRecordDataset, 
      num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.batch(
      batch_size, 
      drop_remainder=True)    # Batch Size

    dataset = dataset.map(
      _parse_function, 
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
    #dataset = dataset.batch(batch_size)    # Batch Size

    dataset = dataset.cache()

    #dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE) #.unbatch()

    #if self.shuffle:
    dataset = dataset.repeat(-1) # Repeat the dataset this time
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


if __name__ == '__main__':
    filepath = os.path.join(os.getcwd(),'DATA','train*.tfrecord')

    tfrecords = list(glob.glob(filepath))
    print(tfrecords)


    dataset = parse_tfrecords(
        filenames=tfrecords, 
        batch_size=2)

    for data in dataset.take(5):
        data_decoded = data.numpy()
        print(data_decoded.shape, data_decoded.dtype)