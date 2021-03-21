import cv2
import numpy as np

def letterbox(img, new_shape=(416, 416), color=(114,114,114), scaleup=True):
    """
    Resize the img to the given shape (new_shape) and calculate the corresponding ratio and padding value

    Input
        img: Input img
        new_shape: input shape for the neural network.
        color: padding color

    Return
        img: image after resize and paddings
        ratio: resize ratio (ratio_x, ratio_y)
        pad: padding (dw, dh)
    """
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    current_shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(float(new_shape[0]) / current_shape[0], float(new_shape[1]) / current_shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(current_shape[1] * r)), int(round(current_shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if current_shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)