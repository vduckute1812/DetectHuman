import os
import cv2 as cv
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Dir path of this file (setting.py)

NAME_LIST = {
    "Person": 1,
    "NonePerson"  : 0,
}
IN_PATH     = os.path.join(BASE_DIR, "data_train")
OUT_PATH    = os.path.join(BASE_DIR, "build_data")
