import math
import os
import numpy as np
import cv2



def get_hog(
    winSize = (50,100),
    blockSize = (10,20),
    blockStride = (5,10),
    cellSize = (10,10),
    nbins = 9,
    derivAperture = 1,
    winSigma = -1.0, 
    histogramNormType = 0,
    L2HysThreshold = 0.2,
    gammaCorrection = 1,
    nlevels = 64,
    signedGradients = True,
    ):

    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,
                            winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradients)
    return hog
    

#Tính các thành phần Gradient của ảnh
def getGradientElement(image, angleMax):
    #image = np.float32(image)/255.0
    sobelX = np.array([[-1, 0, 1],
                       [-2, 0, 2], 
                       [-1, 0, 1]])
    
    sobelY = np.array([ [-1 ,-2, -1],
                        [ 0 , 0,  0],
                        [ 1 , 2,  1]])
    
    x_index, y_index = sobelX.shape
    rows, cols = image.shape
    si = rows, cols, 1
    resultX = np.zeros(si, dtype=np.float32)
    resultY = np.zeros(si, dtype=np.float32)

    x_index = (int)(x_index/2);
    y_index = (int)(y_index/2);

    for x in range(0, rows):
        for y in range(0, cols):
            for u in range(-x_index, x_index+1):
                for v in range(-y_index, y_index+1):
                    vlueX = x-u
                    vlueY = y-v
                    if(vlueX<0 or vlueX>(rows-1) or vlueY<0 or vlueY>(cols-1)):
                        continue
                    resultX[x][y] += sobelX[u+x_index][v+y_index]*image[vlueX][vlueY]
                    resultY[x][y] += sobelY[u+x_index][v+y_index]*image[vlueX][vlueY]       

    result = np.zeros(si, dtype=np.float32)
    angle = np.zeros(si, dtype=np.float32)
    for x in range(0, rows):
        for y in range(0, cols):
            result[x][y]=math.sqrt(math.pow(resultX[x][y],2)+math.pow(resultY[x][y],2))
            if(angleMax==180):
                angle[x][y] = math.fabs(math.atan2(resultY[x][y],resultX[x][y])*180/math.pi)    #0..180
            elif(angleMax==360):
                angle[x][y] = (math.atan2(resultY[x][y],resultX[x][y]))*180/math.pi    #0..360
                if(angle[x][y]<0):
                    angle[x][y]+=360;
            else:
                print("ERROR INPUT ANGLE VALUE!!!")
                sys.exit(0)
    resultX = abs(resultX)
    resultY = abs(resultY)
    resultX=cv2.normalize(resultX,resultX, alpha=0,beta=255, norm_type=cv2.NORM_MINMAX,dtype= cv2.CV_8UC1)
    resultY=cv2.normalize(resultY,resultY, alpha=0,beta=255, norm_type=cv2.NORM_MINMAX,dtype= cv2.CV_8UC1)
    result =cv2.normalize(result ,result , alpha=0,beta=255, norm_type=cv2.NORM_MINMAX,dtype= cv2.CV_8UC1)
    return result, angle, resultX, resultY

#Tạo mảng chứa các thành phần độ lớn của grandient theo hướng
def zeroBin(size):
    arr = np.zeros(size)
    return arr

#Tìm vị trí và tỉ lệ các bin
def binFor(angleCells, bins, maxAngle): 
    angle = int(maxAngle/bins)
    if(angleCells > maxAngle- angle):
        bin1 = int(angleCells/ angle)
        bin2 = 0
        ratio1 = (angleCells - bin1*angle)/ angle
        ratio2 = (maxAngle - angleCells)/ angle
    else:
        bin1 = int(angleCells/ angle)
        bin2 = bin1+1
        ratio1 = (angleCells - bin1* angle)/ angle
        ratio2 = (bin2* angle - angleCells)/ angle
    return bin1, bin2, ratio1, ratio2

#Lấy thành phần Histogram tại (x,y)
def getHistogram(mag, angle, x, y, cellSizes, bins, maxAngle):
    angleBins = maxAngle/bins
    histogram = zeroBin(bins)
    for i in range(cellSizes):
        for j in range(cellSizes):
            dimention = angle[y+i][x+j]
            if(dimention% angleBins == 0):
                if(dimention==maxAngle):
                    dimention=0
                histogram[int(dimention/ angleBins)]+=mag[y+i][x+j]
                continue	     
            bin1, bin2, ratio1, ratio2 = binFor(dimention, bins, maxAngle)
            histogram[bin1] = ratio2*mag[y+i][x+j]
            histogram[bin2] = ratio1*mag[y+i][x+j]
    return histogram    

#Trích xuất Histogram của tất cả các cell
def extractHistograms(img, cellSizes, bins, maxAngle):
    height, width = img.shape
    mag, angle, resultX, resultY = getGradientElement(img, maxAngle)
    cellsWide =  int(width / cellSizes)
    cellsHigh =  int(height / cellSizes)
    size      =  (cellsHigh, cellsWide, bins)
    histograms = np.zeros(size)
    for i in range(cellsWide):
        for j in range(cellsHigh):
            histograms[i][j] = getHistogram(mag, angle, i*cellSizes, j*cellSizes, cellSizes, bins, maxAngle)
    return histograms

def extractHOGFromHistograms(histograms, blockSize, blockStride, normType): #blockStride = 1/2 blockSize
    hight, width, bins = histograms.shape
    blocks = []
    blockHigh = 2* int(hight/(blockSize)) - 1
    blockWide = 2* int(width/(blockSize)) - 1
    for x in range(0, blockWide, blockStride):
        for y in range(0, blockHigh, blockStride):
            block = getBlock(histograms, x, y, blockSize)
            block = normalizeBlock(block, normType)
            blocks.append(block)
    return blocks

#Lấy thành phần của Block
def getBlock(histograms, x, y, length):
    square = []
    for i in range(x, x+length):
        for j in range(y, y+length):
            square.append(histograms[i][j])
    square = np.array(square).flatten()
    return square


def normalizeBlock(block, type):
    size = block.shape[0]
    epsilon = 0.00001
    def L1():
        norm = 0
        for i in range(size):
            norm+=math.fabs(block[i])
        demon = norm + epsilon
        for i in range(size):
            block[i]/=demon
        return block
    def L1_sqrt():
        norm = 0
        for i in range(size):
            norm+=math.fabs(block[i])
        demon = norm + epsilon
        for i in range(size):
            block[i]=math.sqrt(block[i]/demon)
        return block
    def L2():
        sum = 0
        for i in range(size):
            sum+=math.pow(block[i],2)
        demon = math.sqrt(sum+epsilon)
        for i in range(size):
            block[i]/=demon
        return block
    blockNorm = locals()[type]()
    return blockNorm


def getHOGDescriptor(img, cellSize, blockSize, blockStride, normType, bins, angleMax):
    result, angle, resultX, resultY = getGradientElement(img, angleMax)
    histogram = extractHistograms(img, cellSize, bins, angleMax)
    blocks = extractHOGFromHistograms(histogram, int(blockSize/cellSize), int(blockStride/cellSize), normType)
    blocks = np.array(blocks).flatten()
    return blocks
