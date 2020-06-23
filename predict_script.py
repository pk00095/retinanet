from tensorflow import keras
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess_input
from retinanet import get_retinanet_r50, retinanet_bbox
import numpy as np
import cv2


def load_model(checkpoint_path):

    pred_model = keras.models.load_model(
        filepath=checkpoint_path,
        compile=False)

    print('Weights are Loaded ...')

    return pred_model

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
    # pad image
    padded_image = np.zeros(shape=(height, width,3), dtype=image.dtype)
    h,w,_ =  image.shape
    padded_image[:h,:w,:] = image

    # resize image
    resized_image = cv2.resize(padded_image, None, fx=scale, fy=scale).astype(keras.backend.floatx())
    return resized_image


def predict(model, image_list):
    assert isinstance(image_list, list), 'expected a list of images'
    
    images = list()
    h_max, w_max = 0,0

    for img_path in image_list:
        im = np.array(keras.preprocessing.image.load_img(path=img_path))
        h,w, _ = im.shape
        h_max = max(h_max, h)
        w_max = max(w_max, w)
        images.append(im)

    smallest_side = min(h_max, w_max)
    scale = 800 / smallest_side
    largest_side = max(h_max, w_max)
    # scale = tf.cond(largest_side * tf.cast(scale, tf.int32) > 1333, lambda: 1333 / largest_side, , lambda: scale)
    if largest_side * scale > 1333:
        scale = 1333 / largest_side

    images_batch =  list(map(lambda x:pad_resize(x, h_max, w_max, scale), images))


    images_batch = resnet_preprocess_input(np.array(images_batch))

    return model.predict(images_batch)




if __name__ == '__main__':
    pred_model = load_model('./checkpoints/prediction')
    results = predict(pred_model, ['./aerial-vehicles-dataset/images/DJI_0005-0041.jpg'])
    print(results)
