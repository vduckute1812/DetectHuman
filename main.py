import os
import time
import cv2 as cv
import numpy as np
from sklearn.externals import joblib

from helper.__init__ import list_item
from setting import  BASE_DIR, IN_PATH, NAME_LIST
from SVM import estimateParam, resultSVM, SVM_init, createExcelFile, trainTotalData
from HOG import get_hog

num_of_data = {
    "Person": 0,
    "NonePerson": 0,
}

hog = get_hog()

def description(input_path):
    X = []
    y = []
    timePath = []
    pathItem = []
    for idx, path_img in enumerate(list_item(input_path)) :
        name = os.path.split(os.path.dirname(path_img))[1]
        image = cv.imread(path_img, cv.IMREAD_GRAYSCALE)
        num_of_data[name] += 1
        # Getting the time for extracting a hog feature
        t = time.clock()
        hog_value = hog.compute(image)
        timeProcessing = time.clock()-t

        X.append(hog_value.flatten())
        y.append(NAME_LIST[name])
        timePath.append(timeProcessing)
        pathItem.append(path_img)

    return X, y, num_of_data, pathItem, timePath


X, y, num_of_data, item_path, timePath = description(IN_PATH)
# estimateParam(X, y)

svm = SVM_init(10, 0.1, 'rbf')

nameExcel = "resultSVM.xls"
createExcelFile(nameExcel)

resultSVM(X,y,num_of_data, item_path, svm, nameExcel, timePath, "recognizingTrue", "recognizingFalse")
