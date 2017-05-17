'''This module contains several common helper functions and classes.
These helpers are used throught the modules in the CarWorld package
'''
import cv2
import numpy as np


def overlay_img(orig, update):
    '''Overlay an image over another image'''
    # If one of the images is gray scale and the other is RGB, convert to RGB
    if len(orig.shape) > len(update.shape):
        update = cv2.cvtColor(update, cv2.COLOR_GRAY2RGB)
    elif len(orig.shape) < len(update.shape):
        orig = cv2.cvtColor(orig, cv2.COLOR_GRAY2RGB)
    return cv2.addWeighted(orig, 1, update, 0.3, 0)

def draw_boxes(img, boxpts, color=(0, 0, 255), thick=6):
    '''Given an image and list of topleft/bottomright box pts, draw boxes'''
    assert img is not None
    if len(boxpts) <= 0:
        return img

    draw_img = np.copy(img) # XXX: Is this necessary?
    for bbox in boxpts:
        cv2.rectangle(draw_img, bbox[0], bbox[1], color, thick)
    return draw_img

def convert_img(img, dst, src='BRG'):
    '''Convert an img from format src to format dst'''
    if src == 'BRG':
        color_map = {
                'HSV': cv2.COLOR_BGR2HSV,
                'LUV':cv2.COLOR_BGR2LUV,
                'HLS':cv2.COLOR_BGR2HLS,
                'YUV':cv2.COLOR_BGR2YUV,
                'YCrCb':cv2.COLOR_BGR2YCrCb,
                'GRAY': cv2.COLOR_BGR2GRAY
        }
    elif src == 'RGB':
        color_map = {
                'HSV': cv2.COLOR_RGB2HSV,
                'LUV':cv2.COLOR_RGB2LUV,
                'HLS':cv2.COLOR_RGB2HLS,
                'YUV':cv2.COLOR_RGB2YUV,
                'YCrCb':cv2.COLOR_RGB2YCrCb,
                'GRAY': cv2.COLOR_RGB2GRAY
        }
    else:
        raise NotImplementedError("Cannot convert from source colour. %s" %src)

    # Convert image to new color space (if specified)
    if dst in color_map:
        feature_image = cv2.cvtColor(img, color_map[dst])
    else:
        feature_image = np.copy(img)
    return feature_image
