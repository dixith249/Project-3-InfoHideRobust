#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import cv2
import imutils
import numpy as np
from datetime import datetime

def rotation(image, angle):
    rotated = imutils.rotate_bound(image, angle)
    return rotated

def translation(image):    
    height, width = image.shape[:2]
    tX = np.random.randint(low=(-height+1)/2, high=(height+1)/2)     
    tY = np.random.randint(low=(-width+1)/2, high=(width-1)/2)
    T = np.float32([[1, 0, int(tX)], [0, 1, int(tY)]])   
    translated = cv2.warpAffine(image, T, (width, height))
    return translated

def scaling(image, scalePercent):
    height, width = image.shape[:2]
    height = int(height * scalePercent / 100)
    width = int(width * scalePercent / 100)
    resized = cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)
    return resized

def cropping(image, cropPercent):
    height, width = image.shape[:2]
    heightCrop = int(height * cropPercent / 100)
    widthCrop = int(width * cropPercent / 100)
    yStart = heightCrop
    yEnd = height - heightCrop
    xStart = widthCrop
    xEnd = width - widthCrop
    cropped = image[yStart:yStart+yEnd, xStart:xStart+xEnd]
    return cropped

def histogramEqualization(image):
    imageHSV =  cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    imageV = imageHSV[:, :, 2]
    imageHSV[:, :, 2] = cv2.equalizeHist(imageV)
    equalizedImage = cv2.cvtColor(imageHSV, cv2.COLOR_HSV2BGR)
    return equalizedImage

def sharpening(image):
    kernelSharpening = np.array([[-1,-1,-1], 
                              [-1, 9,-1],
                              [-1,-1,-1]])
    sharpened = cv2.filter2D(image, -1, kernelSharpening)
    return sharpened

def edgeDetection(image, edgeType):
    if edgeType == 6:
        kernelEdge = np.array([[-1,-1,-1], 
                              [-1, 8,-1],
                              [-1,-1,-1]])
    elif edgeType == 7:
        kernelEdge = np.array([[1,0,-1], 
                              [0, 0,0],
                              [-1,0,1]])
    elif edgeType == 8:
        kernelEdge = np.array([[0,1,0], 
                              [1, -4,1],
                              [0,-1,0]])
        
    edges = cv2.filter2D(image, -1, kernelEdge)
    return edges

def saltAndPepperNoise(image, a = 0.01, b = 0.99):
    height, width = image.shape[0], image.shape[1]
    for i in range(height):
        for j in range(width):
            thresh = np.random.random()
            if thresh <= a:
                image[i][j] = [0,0,0] 
            elif thresh >= b:
                image[i][j] = [255,255,255]
    return image

def gaussianNoise(image, mean = 0, sigma = 60):
    noised = image + np.random.normal(mean, sigma, image.shape)
    return noised

def uniformNoise(image, a = 1, b = 60):
    noised = image + np.random.uniform(a, b, image.shape)
    return noised

def erlangNoise(image, a = 1, b = 60):
    noised = image + np.random.gamma(a, b, image.shape)
    return noised

def exponentialNoise(image, a = 60):
    noised = image + np.random.exponential(a, image.shape)
    return noised

def rayleighNoise(image, a = 60):
    noised = image + np.random.rayleigh(a, image.shape)
    return noised

def poissonNoise(image, mean = 60):
    noised = image + np.random.poisson(mean, image.shape)
    return noised

def blurAverage(image, kernelSize=5):
    blurred = cv2.blur(image,(kernelSize,kernelSize))
    return blurred

def blurGaussian(image, kernelSize=5, stdDev = 0):
    if kernelSize % 2 == 0:
        kernelSize = kernelSize + 1
    blurred = cv2.GaussianBlur(image,(kernelSize,kernelSize), stdDev)
    return blurred

def blurMedian(image, kernelSize=5):
    if kernelSize % 2 == 0:
        kernelSize = kernelSize + 1
    blurred = cv2.medianBlur(image,kernelSize)
    return blurred

def blurBilateral(image, diameter=9, sigmaColor=75, sigmaSpace=75):
    blurred = cv2.bilateralFilter(image,diameter, sigmaColor, sigmaSpace)
    return blurred

def selectOption(option):
    if option < 0 or option > 24:
        print("Invalid option. Moving on to the next.")
    elif option == 20:
        for actualOption in range(6,9):
            selectOption(actualOption)
    elif option == 21:
        for actualOption in range(9,16):
            selectOption(actualOption)
    elif option == 22:
        for actualOption in range(16,20):
            selectOption(actualOption)
    elif option == 23:
        for actualOption in range(0,20):
            selectOption(actualOption)
    else:
        pathSave = os.path.join(outputDir,timestamp,attacks[option])
        print("")
        verifDirExist = os.path.isdir(pathSave)
        if verifDirExist == False:
            os.mkdir(os.path.join(outputDir,timestamp,attacks[option]))
        print(str(option)+" - "+ attacks[option])
        if option == 0:
            angle = input("Please enter the angle of rotation(integer values):")
            angle = int(angle)
            for imageName, image in zip(imageNames, imageList):
                imageName = os.path.splitext(imageName)[0]
                rotatedImage = rotation(image, angle)
                cv2.imwrite(pathSave+"/"+imageName+"Angle"+str(angle)+".png", rotatedImage)
        
        if option == 1:
            for imageName, image in zip(imageNames, imageList):
                imageName = os.path.splitext(imageName)[0]
                translatedImage = translation(image)
                cv2.imwrite(pathSave+"/"+imageName+".png", translatedImage)
        
        if option == 2:
            percent = input("Please enter the percent of scaling(integer postive values):")
            percent = int(percent)
            for imageName, image in zip(imageNames, imageList):
                imageName = os.path.splitext(imageName)[0]
                scaledImage = scaling(image, percent)
                cv2.imwrite(pathSave+"/"+imageName+"Scale"+str(percent)+".png", scaledImage)
                
        if option == 3:
            percent = input("Please enter the percent of crop(integer postive values 0-100):")
            percent = int(percent)
            for imageName, image in zip(imageNames, imageList):
                imageName = os.path.splitext(imageName)[0]
                croppedImage = cropping(image, percent)
                cv2.imwrite(pathSave+"/"+imageName+"Crop"+str(percent)+".png", croppedImage)
        
        if option == 4:
            for imageName, image in zip(imageNames, imageList):
                imageName = os.path.splitext(imageName)[0]
                histEqImage = histogramEqualization(image)
                cv2.imwrite(pathSave+"/"+imageName+".png", histEqImage)
        
        if option == 5:
            for imageName, image in zip(imageNames, imageList):
                imageName = os.path.splitext(imageName)[0]
                sharpImage = sharpening(image)
                cv2.imwrite(pathSave+"/"+imageName+".png", sharpImage)
                
        if option == 6 or option == 7 or option == 8:
            for imageName, image in zip(imageNames, imageList):
                imageName = os.path.splitext(imageName)[0]
                edgeImage = edgeDetection(image, option)
                cv2.imwrite(pathSave+"/"+imageName+".png", edgeImage)
                
        if option == 9:
            for imageName, image in zip(imageNames, imageList):
                imageName = os.path.splitext(imageName)[0]
                noisedImage = saltAndPepperNoise(image)
                cv2.imwrite(pathSave+"/"+imageName+".png", noisedImage)
            
        if option == 10:
            for imageName, image in zip(imageNames, imageList):
                imageName = os.path.splitext(imageName)[0]
                noisedImage = gaussianNoise(image)
                cv2.imwrite(pathSave+"/"+imageName+".png", noisedImage)
                
        if option == 11:
            for imageName, image in zip(imageNames, imageList):
                imageName = os.path.splitext(imageName)[0]
                noisedImage = uniformNoise(image)
                cv2.imwrite(pathSave+"/"+imageName+".png", noisedImage)
                
        if option == 12:
            for imageName, image in zip(imageNames, imageList):
                imageName = os.path.splitext(imageName)[0]
                noisedImage = rayleighNoise(image)
                cv2.imwrite(pathSave+"/"+imageName+".png", noisedImage)
                
        if option == 13:
            for imageName, image in zip(imageNames, imageList):
                imageName = os.path.splitext(imageName)[0]
                noisedImage = poissonNoise(image)
                cv2.imwrite(pathSave+"/"+imageName+".png", noisedImage)
                
        if option == 14:
            for imageName, image in zip(imageNames, imageList):
                imageName = os.path.splitext(imageName)[0]
                noisedImage = erlangNoise(image)
                cv2.imwrite(pathSave+"/"+imageName+".png", noisedImage)
                
        if option == 15:
            for imageName, image in zip(imageNames, imageList):
                imageName = os.path.splitext(imageName)[0]
                noisedImage = exponentialNoise(image)
                cv2.imwrite(pathSave+"/"+imageName+".png", noisedImage)

        if option == 16:
            kernelSize = input("Please enter the kernel size(integer postive values):")
            kernelSize = int(kernelSize)
            for imageName, image in zip(imageNames, imageList):
                imageName = os.path.splitext(imageName)[0]
                blurredImage = blurAverage(image, kernelSize)
                cv2.imwrite(pathSave+"/"+imageName+".png", blurredImage)
        
        if option == 17:
            kernelSize = input("Please enter the kernel size(integer postive values):")
            kernelSize = int(kernelSize)
            for imageName, image in zip(imageNames, imageList):
                imageName = os.path.splitext(imageName)[0]
                blurredImage = blurGaussian(image, kernelSize)
                cv2.imwrite(pathSave+"/"+imageName+".png", blurredImage)
        
        if option == 18:
            kernelSize = input("Please enter the kernel size(integer postive values):")
            kernelSize = int(kernelSize)
            for imageName, image in zip(imageNames, imageList):
                imageName = os.path.splitext(imageName)[0]
                blurredImage = blurMedian(image, kernelSize)
                cv2.imwrite(pathSave+"/"+imageName+".png", blurredImage)
                
        if option == 19:
            diameter = input("Please enter the diameter size(integer postive values):")
            diameter = int(diameter)
            for imageName, image in zip(imageNames, imageList):
                imageName = os.path.splitext(imageName)[0]
                blurredImage = blurBilateral(image, diameter)
                cv2.imwrite(pathSave+"/"+imageName+".png", blurredImage)

attacks = ["Rotation", "Translation", "Scaling", "Cropping", 
           "Histogram equalization", "Sharpening", 
           "Edge detection 1", "Edge detection 2","Edge detection 3",
           "Noise - Salt & Pepper", "Noise - Gaussian", "Noise - Uniform",
           "Noise - Rayleigh", "Noise - Poisson",
           "Noise - Erlang", "Noise - Exponential",
           "Blur - Average", "Blur- Gaussian", 
           "Blur - Median", "Blur - Bilateral", 
           "All edge detection", "All noise", "All blur", "All attacks"]

inputDir = "Input"
outputDir = "Output"
imageNames = []
imageList = []
validImages = [".jpeg", ".jpg",".gif",".png", ".bmp"]

verifDirExist = os.path.isdir(inputDir)
if verifDirExist == False:
    os.mkdir(inputDir)
    
verifDirExist = os.path.isdir(outputDir)
if verifDirExist == False:
    os.mkdir(outputDir)

for file in os.listdir(inputDir):
    ext = os.path.splitext(file)[1]
    if ext.lower() not in validImages:
        continue
    else:
        imageNames.append(file)
        imageList.append(cv2.imread(os.path.join(inputDir,file)))
                
if not imageList:
    raise Exception("Sorry, no images on input folder. Please place images on the specified folder: "+inputDir)
    
now = datetime.now()
timestamp = now.strftime("%d%m%Y%H%M%S")
os.mkdir(os.path.join(outputDir,timestamp))

print("Attacks to image")
for i in range(len(attacks)):
    print(i,"-",attacks[i])
    
operations = input("Please enter the number of options of the attack spaced:")
operations = operations.split(" ")

for i in range(0, len(operations)): 
    if operations[i].isdigit() == True:
        operations[i] = int(operations[i])
    else:
        operations[i] = -1

for operation in operations:
    selectOption(operation)


