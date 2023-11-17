import os
import cv2
import numpy as np
import base64
from segment_anything import sam_model_registry, SamPredictor
import torch

# helper function that loads an image before adding it to the widget
def encode_image(filepath):
    with open(filepath, 'rb') as f:
        image_bytes = f.read()
    encoded = str(base64.b64encode(image_bytes), 'utf-8')
    return "data:image/jpg;base64,"+encoded


def generate_action_mask(sam, img_path, action_class_id, bbox):
    """
    Generates a simple rectangular mask for a given action class ID and bounding box.

    :param action_class_id: The class ID of the action.
    :param bbox: A tuple of (x_min, y_min, x_max, y_max) for the bounding box.
    :param mask_size: A tuple of (width, height) representing the size of the mask.
    :return: A binary mask with the action area filled.
    """
    
    
    mask_predictor = SamPredictor(sam)
    
    IMAGE_PATH = img_path
    
    box = np.array([
        bbox[0],
        bbox[1],
        bbox[2],
        bbox[3]
    ])
    image_bgr = cv2.imread(IMAGE_PATH)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    mask_predictor.set_image(image_rgb)

    masks, scores, logits = mask_predictor.predict(
        box=box,
        multimask_output=False
    )
    
    mask = masks[0]
    # Convert the mask to a color image (green for True, black for False)
    mask_color = np.zeros((mask.shape[0], mask.shape[1]), dtype='uint8')  # Initialize a black image
    mask_color[mask] = 255  # Set True pixels to green
    
    # Save the array as an image
    # mask = np.expand_dims(mask, axis = -1)
    # mask = np.concatenate((mask,mask,mask),axis=-1)

    return mask
