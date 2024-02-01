import random


def generate_rgb_color():
    return tuple([random.randint(0, 255) for _ in range(3)])


TRACKING_LEN = 25
