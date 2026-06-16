"""Shared synthetic-image helpers for redboxflip tests."""
import cv2
import numpy as np


def make_redbox_image(w=600, h=400, box=(80, 60, 520, 340), case=(120, 100, 480, 300)):
    """White sheet, a red rectangle outline, and a dark 'case' filling most of it."""
    img = np.full((h, w, 3), 255, np.uint8)
    bx0, by0, bx1, by1 = box
    cv2.rectangle(img, (bx0, by0), (bx1, by1), (0, 0, 255), 6)   # red (BGR)
    cx0, cy0, cx1, cy1 = case
    cv2.rectangle(img, (cx0, cy0), (cx1, cy1), (40, 30, 30), -1)  # dark case
    return img
