import random

import cv2
import numpy as np


def generate_rgb_color() -> tuple:
    return tuple([random.randint(0, 255) for _ in range(3)])


def is_trespassing(x, y) -> bool:
    min_p, max_p = TRESPASS_COORDS
    return min_p[0] < x < max_p[0] and min_p[1] < y < max_p[1]


def draw_trespass_region(frame: np.ndarray) -> np.ndarray:
    min_p, max_p = TRESPASS_COORDS
    sub_img = frame[min_p[0]:max_p[0], min_p[1]:max_p[1]]
    red_rect = np.full(sub_img.shape, fill_value=[0, 0, 255], dtype=np.uint8)
    res = cv2.addWeighted(sub_img, 0.8, red_rect, 0.5, 1.0)
    frame[min_p[0]:max_p[0], min_p[1]:max_p[1]] = res

    return frame


TRACKING_LEN = 25
TRESPASS_COORDS = [[220, 220], [440, 440]]
