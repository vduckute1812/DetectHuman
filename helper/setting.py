import os
import cv2 as cv

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Dir path of this file (setting.py)

NAME_LIST = {
    "Human": 'cos',
    "None Human": "ant",
}

OUT_PATH    = os.path.join(BASE_DIR, "train_data")
