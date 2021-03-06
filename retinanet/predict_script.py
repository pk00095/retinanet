"""Summary
"""
from tensorflow import keras
from tensorflow.keras.applications.resnet import preprocess_input as resnet_normalize_image
import numpy as np
import cv2

from PIL import Image, ImageDraw, ImageFont

font = ImageFont.load_default()

interpolation_options = {
    'nearest':cv2.INTER_NEAREST,
    'linear':cv2.INTER_LINEAR,
    'cubic':cv2.INTER_CUBIC,
    'area':cv2.INTER_AREA,
    'lanczos4':cv2.INTER_LANCZOS4
}


STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


def load_model(checkpoint_path):
    """Summary
    
    Args:
        checkpoint_path (string): path to prediction model checkpoint
    
    Returns:
        tf.keras.model.Model: The prediction model
    """
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


def predict(model, image_path, min_side=800, max_side=1333):
    """takes an image, passes through model and returns bboxes, scores, labels
    
    Args:
        model (tf.keras.model.Model): the prediction model object
        image_path (str): path to image to run prediction on
        min_side (int, optional): minimum dimension, defaults to 800
        max_side (int, optional): maximum dimension, defaults to 1333
    
    Returns:
        TYPE: bboxes(x1,y1,x2,y2), confidence, labels
    """
    images = list()
    h_max, w_max = 0,0

    image_list = [image_path]

    for img_path in image_list:
        im = np.array(keras.preprocessing.image.load_img(path=img_path))
        h,w, _ = im.shape
        h_max = max(h_max, h)
        w_max = max(w_max, w)
        images.append(im)

    smallest_side = min(h_max, w_max)
    scale = min_side / smallest_side
    largest_side = max(h_max, w_max)
    # scale = tf.cond(largest_side * tf.cast(scale, tf.int32) > 1333, lambda: 1333 / largest_side, , lambda: scale)
    if largest_side * scale > max_side:
        scale = max_side / largest_side

    images_batch =  list(map(lambda x:pad_resize(x, h_max, w_max, scale), images))

    images_batch = resnet_normalize_image(np.array(images_batch))

    bbox, confidence, label = model.predict(images_batch)

    return bbox[0].astype(int)/scale, confidence[0], label[0], im

def annotate_image(image_path, bboxes, scores, labels, threshold=0.5, label_dict=None):
  """Summary
  
  Args:
      image_path (str): path to image to annotate
      bboxes (TYPE): Description
      scores (TYPE): Description
      labels (TYPE): Description
      threshold (float, optional): Description
      label_dict (None, optional): Description
  
  Returns:
      TYPE: Description
  """
  image = Image.open(image_path)
  Imagedraw = ImageDraw.Draw(image)

  for box, label, score in zip(bboxes, labels, scores):
    if score < threshold:
      continue

    (left,top,right,bottom) = box

    label_to_display = label
    if isinstance(label_dict, dict):
      label_to_display = label_dict[label]

    caption = "{}|{:.3f}".format(label_to_display, score)
    #draw_caption(draw, b, caption)

    colortofill = STANDARD_COLORS[label]
    Imagedraw.rectangle([left,top,right,bottom], fill=None, outline=colortofill)

    display_str_heights = font.getsize(caption)[1]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * display_str_heights

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height

    text_width, text_height = font.getsize(caption)
    margin = np.ceil(0.05 * text_height)
    Imagedraw.rectangle([(left, text_bottom-text_height-2*margin), (left+text_width,text_bottom)], fill=colortofill)

    Imagedraw.text((left+margin, text_bottom-text_height-margin),caption,fill='black',font=font)

  return image



if __name__ == '__main__':
    pred_model = load_model('./checkpoints/prediction')
    bbox, confidence, label, im = predict(pred_model, './aerial-vehicles-dataset/images/DJI_0005-0041.jpg')
    # import pdb; pdb.set_trace()
    # print(bbox.shape, bbox.shape, confidence.shape, label.shape, im.shape)

    annotated_image = annotate_image(
        image_array=im, 
        bboxes=bbox, 
        scores=confidence, 
        labels=label, 
        threshold=0.5, 
        label_dict=None)

    annotated_image.save('annotated.jpg')
