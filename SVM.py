from __future__ import print_function
from setting import  BASE_DIR, IN_PATH, NAME_LIST

import math
import os
import cv2
import numpy as np
from tempfile import TemporaryFile
from xlwt import Workbook,easyxf,Style
from xlrd import open_workbook
from xlutils.copy import copy
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

label = {
    "NonePerson": 0,
    "Person": 1
}

get_name = {
    0: "NonePerson",
    1: "Person"
}

def SVM_init(C, gamma, kernel):
    svm = SVC(C=C, kernel=kernel, gamma=gamma)
    return svm


def resultSVM(X, y, num_of_flower, pathList,svm, nameFile, timeExecute, outTrueFile, outFalseFile):
    percentData = []
    item = 0
    item2 = 0
    num=0
    result = []
    table = []
    table2 = []
    nameItemWrite = []
    
    ResultDirectory = "ResultRecognizing"
    size = len(NAME_LIST)

    for x in range(size):
        detectTable = [0]*len(label)
        nameItem = get_name[y[num]]
        start = num
        num += num_of_flower[nameItem]
        listPathFlower = pathList[start:num]
        #num+=200
        print(">>>>Item : ", nameItem)
        directory = os.path.join(BASE_DIR, ResultDirectory, nameItem)
        if not os.path.exists(directory):
            os.makedirs(directory)    
        del result[:]
        del table [:]
        del table2 [:]
        for index in range(start, num, int(num_of_flower[nameItem]/10)):
            fileName = os.path.join(directory,str(index)+".pkl")
            s = index
            f = index+int(num_of_flower[nameItem]/10) if index+int(num_of_flower[nameItem]/10)<num else num
            inTest = X[s:f]
            inTrain = X[:s] + X[f:]
            labelTrain = y[:s] + y[f:]
            # svm.fit(inTrain, labelTrain) 
            # joblib.dump(svm, fileName)

            print(index)
            svm = joblib.load(fileName)
            predict = svm.predict(inTest)
            result.append(list(predict.flatten()))
            list_value = calculate(list((predict.flatten())), size)
            table.append(list_value)
            for lb_result in predict:
                detectTable[lb_result]+=1
                table2.append(lb_result)
        
        detectTable = map(lambda x:  x*100/num_of_flower[nameItem],detectTable)
        percentData.append(list(detectTable))
        nameItemWrite.append(nameItem)
        
        saveFile(table, nameItem, num_of_flower[nameItem], item, 0, nameFile)    
        item += len(table)+5
        table2=[table2]
        saveFile2(table2, nameItem, num_of_flower[nameItem],listPathFlower, item2, 1, nameFile, outTrueFile,outFalseFile, timeExecute[start:num])
        item2 += len(table2) + 5
    print("percent detail")
    saveFile3(percentData,nameItemWrite,2, nameFile)
    print(percentData)

def trainTotalData(X, y, svm):

    ResultDirectory = "AllRecognizing"

    directory = os.path.join(BASE_DIR, ResultDirectory)
    if not os.path.exists(directory):
        os.makedirs(directory)    

    fileName = os.path.join(directory,"all_data"+".pkl")
    svm.fit(X, y) 
    joblib.dump(svm, fileName)


def detailResult(X,y,num_of_flower,svm):
    percentData = []
    nameItemWrite = []
    size = len(label)
    ResultDirectory = "hogTraining"
    item2 = 0
    num=0
    result2 = []
    table2 = []
    for x in range(size):
        nameItem = get_name[y[num]]
        start = num
        num += num_of_flower[nameItem]
        print(">>>>Item : ", nameItem)
        directory = os.path.join(BASE_DIR, ResultDirectory, nameItem)

        if not os.path.exists(directory):
           os.makedirs(directory)    
        
        # reset data
        del result2[:]
        del table2 [:]
        detectTable = [0, 0]

        for index in range(start, num):
            fileName = os.path.join(directory,str(index)+".pkl")
            inTest = X[index]
            inTrain = X[:index] + X[index+1:]
            labelTrain = y[:index] + y[index+1:]
            svm.fit(inTrain, labelTrain)
            joblib.dump(svm, fileName)
            print(index)


            svm = joblib.load(fileName)
            predict = svm.predict([inTest])

            result2.append(predict)
            detectTable[predict[0]] += 1
        table2.append(result2)
        detectTable = map(lambda x:  x*100/num_of_flower[nameItem],detectTable)
        percentData.append(list(detectTable))
        nameItemWrite.append(nameItem)

    print("Percent detail")   
    saveFile3(percentData,nameItemWrite,2)
    print(percentData)


def estimateParam(X, y):
    # Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)

    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1,1e-1, 1e-2, 1e-3, 1e-4,1e-5],
                         'C': [1,10,100,1000,10000, 100000]}]

    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
                           scoring='%s_macro' % score)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()

def estimateParam2(X,y):
    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
    grid.fit(X, y)

    print("The best parameters are %s with a score of %0.2f"
          % (grid.best_params_, grid.best_score_))


def calculate(list_value, item):
    buckets = [0]*item
    for i in (list_value):
        buckets[i]+=1
    return buckets

def createExcelFile(nameFile):
    w = Workbook()
    ws = w.add_sheet('SVM')
    ws2 = w.add_sheet('Detail')
    ws3 = w.add_sheet('percentDetail')
    w.save(nameFile)
    w.save(TemporaryFile())


def saveFile(data, nameFlower, numFlower, item, index, nameFile):

    style = easyxf(
    'pattern: pattern solid, fore_colour red;'
    'align: vertical center, horizontal center;'
    )
    style2 = easyxf(
    'align: vertical center, horizontal center;'
    )

    rb = open_workbook(nameFile,formatting_info=True)
    w = copy(rb)
    ws = w.get_sheet(index)
         
    #w = Workbook()
    #ws = w.add_sheet('SVM')
    ws.write_merge(item,item,0,len(data[0]),str(label[nameFlower]+1)+': '+nameFlower,style)

    for i in range (len(data[0])):
         ws.row(item+1).write(i+1, i+1, style2)
    for i in range (len(data)):
         ws.row(item+i+2).write(0, i+1, style2) 

    for i in range (len(data)):
        for j in range (len(data[0])):
            ws.write(item+i+2, j+1, data[i][j])

    num = 0
    for i in range (len(data)):
        num+=data[i][label[nameFlower]]
    ws.write(item+len(data)+2, 0, str(num/numFlower))

    w.save(nameFile)
    w.save(TemporaryFile())


def saveFile2(data, nameFlower, numFlower, pathList, item, index, nameFile, outTrueFile, outFalseFile, timeExecute):
    style = easyxf(
    'pattern: pattern solid, fore_colour red;'
    'align: vertical center, horizontal center;'
    )
    style1 = easyxf(
    'align: vertical center, horizontal center;'
     )

    style2 = easyxf(
    'pattern: pattern solid, fore_colour green;'
    'align: vertical center, horizontal center;'
    )

    style3 = easyxf(
    'pattern: pattern solid, fore_colour blue;'    
    'align: vertical center;'
    )

    rb = open_workbook(nameFile,formatting_info=True)
    w = copy(rb)
    ws = w.get_sheet(index)
    # ws.write(item, 0, nameFlower, style3)
    # # for i in range (len(data[0])):
    # #      ws.row(item+2).write(i, i+1, style1)
    # for i in range (10):
    #      ws.row(item+2).write(i, i+1, style1)
    
    numTrue = 0
    numWrong = 0

    for i in range (len(data)):
        for j in range (len(data[0])):
            if(data[i][j] == label[nameFlower]):
                numTrue+=1
                # ws.write(item+i+3, j, str(data[i][j]), style2)
                # ws.write(item+i+4, j, str(timeExecute[j]))
                value = data[i][j]
                # value = value[0]
                nameFlow = get_name[value]
                nameFlow = os.path.split(os.path.dirname(pathList[j]))[1]
                name = os.path.basename(pathList[j])
                name = name[1:3]+str(numTrue)+'.jpg'
                orgImg = cv2.imread(pathList[j], cv2.IMREAD_GRAYSCALE)
                pathSave = os.path.join(outTrueFile, 'count', nameFlow) 
                if not os.path.exists(pathSave):
                   os.makedirs(pathSave)
                pathSave = os.path.join(pathSave, name)    
                cv2.imwrite(pathSave,orgImg)                    
                                
            else:
                numWrong+=1
                # ws.write(item+i+3, j, str(data[i][j]), style)
                # ws.write(item+i+4, j, str(timeExecute[j]))
                value = data[i][j]
                # value = value[0]
                nameFlow = get_name[value]
                nameFlow = os.path.split(os.path.dirname(pathList[j]))[1]
                name = os.path.basename(pathList[j])
                name = name[1:3]+str(numWrong)+'.jpg'
                orgImg = cv2.imread(pathList[j], cv2.IMREAD_GRAYSCALE)
                pathSave = os.path.join(outFalseFile, 'count', nameFlow) 
                if not os.path.exists(pathSave):
                   os.makedirs(pathSave)
                pathSave = os.path.join(pathSave,name)
                cv2.imwrite(pathSave,orgImg)                    

    average = 0
    for timeEx in timeExecute:
        average += timeEx
    average = average/len(data[0])
    ws.write(item+1, 0, str(numTrue/numFlower))
    ws.write(item+1, 2, str(average))
    w.save(nameFile)
    w.save(TemporaryFile())
        

def saveFile3(data, listNameFlower, index, nameFile):
    rb = open_workbook(nameFile,formatting_info=True)
    w = copy(rb)
    ws = w.get_sheet(index)

    for i in range (len(label)):
         ws.row(1).write(i+1,str(get_name[i]))

    for i in range (len(label)):
         ws.row(i+2).write(0,str(get_name[i]))

    for i in range (len(data)):
        for j in range (len(data[0])):
            ws.write(label[listNameFlower[i]]+2, j+1, str(data[i][j]))

    w.save(nameFile)
    w.save(TemporaryFile())
