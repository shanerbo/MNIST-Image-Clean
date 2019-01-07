
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import os
import re
import glob
from enum import Enum
import shutil

''''
This script will delete all files which are not 1's or 2's by reading csv file and compare filename with the corresponding value in csv file. Ultimately, rename them e.g Ox.jpg
''''

# In[ ]:


def CleanMnistImage(Zero = 0, One = 0, Two = 0, Three = 0, Four = 0, Five = 0, Six = 0, 
                    Seven = 0, Eight = 0, Nine = 0, Train = 0, Test = 0):
    i = 1
    label = []
    if Train ==1 or Test == 1:
        prefix, df1, iterator, Reg, targetFolder = TrainOrTest(Train,Test, zero = Zero, one = One, two = Two, three = Three,
                                                        four = Four, five = Five, six = Six, seven = Seven, eight = Eight, 
                                                         nine = Nine)
        print("Wait...")
        for filename in iterator:
            m = re.search(Reg, filename)
            num = int(m.group(1))
            if (num not in df1.index):
                continue
            else:
                args = locals()
                argument = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]                
                for j in range(10):
                    if df1.loc[num, 'value'] == j and args[argument[j]] == 1:
                        newName = prefix + str(i) + ".jpg"
                        shutil.copy2(filename, targetFolder + newName)
#                         os.rename(filename, targetFolder + newName,)
                        label.append(j)
                        i+=1
        np.savetxt(targetFolder + 'labels.txt', np.array(label)[None], fmt="%d", delimiter=",")
        generateTextFile(label, metadataList, targetFolder, i-1)
        print("Done")
        return i
    else:
        quit(0)
        
def generateTextFile(LabelList, MetadataList, FolderName, ImageCount):
    SampleCount = "SAMPLES_COUNT:"+str(ImageCount)
    MetadataList.insert(2, SampleCount)
    np.savetxt(FolderName + 'labels.txt', np.array(LabelList)[None], fmt="%d", delimiter=",")
    np.savetxt(FolderName + 'metadata.txt', np.array(MetadataList), fmt="%s")
        
def emilinateDigit(value, zero = 0, one = 0, two = 0, three = 0, four = 0, five = 0, six = 0, seven = 0, eight = 0, nine = 0):
    digitList = [zero, one, two, three, four, five, six, seven, eight, nine]
    targetList = []
    for i in range(len(digitList)):
        if digitList[i] == 1:
            targetList.append(i)
    if value in targetList:
        return 1
    else:
        return 0

def TrainOrTest(Train, Test, zero = 0, one = 0, two = 0, three = 0, four = 0, five = 0, six = 0, seven = 0, eight = 0, nine = 0):
    column_names = ['path','value']
    if Train == 1:
        prefix = 'O'
        df = pd.read_csv("train-labels.csv",names = column_names)
        df['bool'] = df.apply(lambda row: emilinateDigit(row['value'], zero = zero, one = one, two = two, three = three,
                                                        four = four, five = five, six = six, seven = seven, eight = eight, 
                                                         nine = nine), axis=1)
        df1 = df.drop(df[df['bool'] == 0].index)
        iterator = glob.glob('train-images//*.jpg')
        Reg = r"train-images\\(\d+)"
        targetFolder = trainFolder+"//"
    elif Test == 1:
        prefix = 'T'
        df = pd.read_csv("test-labels.csv",names = column_names)
        df['bool'] = df.apply(lambda row: emilinateDigit(row['value'], zero = zero, one = one, two = two, three = three,
                                                        four = four, five = five, six = six, seven = seven, eight = eight, 
                                                         nine = nine), axis=1)
        df1 = df.drop(df[df['bool'] == 0].index)
        iterator = glob.glob('test-images//*.jpg')
        Reg = r"test-images\\(\d+)"
        targetFolder = testFolder+"//"
    return prefix, df1, iterator, Reg, targetFolder

def combineMetaDataList(DataSetName, SamplesIn, SampleRow, SampleCol, SamplePrefix, SampleSuffix, ClassesCount, BatchesCount):
    args = locals()
    argv = []
    for key, item in args.items():
        argv.insert(0, item)
    return argv

# In[ ]:


trainFolder = input("Folder Name where all Training data images will go to: ")
testFolder = input("Folder Name where all Testing data images will go to: ")
parameterList = [0,0,0,0,0,0,0,0,0,0]
metadataList = []
SAMPLES_IN = "SAMPLES_IN:Files"
CLASSES_COUNT = SAMPLES_COUNT = 0
SAMPLE_SUFFIX = "SAMPLE_SUFFIX:jpg"
BATCHES_COUNT = input("BATCHES_COUNT: ")
BATCHES_COUNT = "BATCHES_COUNT:"+BATCHES_COUNT
SAMPLE_ROWS = "SAMPLE_ROWS:28"
SAMPLE_COLS = "SAMPLE_COLS:28"
for i in range(len(parameterList)):
    question = "Do you want to extract " + "'" + str(i) + "'" + ": "
    parameterList[i] = int(input(question, ))
    if parameterList[i] == 1:
        CLASSES_COUNT += 1
CLASSES_COUNT = "CLASSES_COUNT:" + str(CLASSES_COUNT)
isTrain = int(input("Generate Training data?:"))
isTest = int(input("Generate Testing data?:"))
# trainFolder = "testTrain"
# testFolder = "testTest"
if not os.path.exists(trainFolder):
    os.makedirs(trainFolder)
if not os.path.exists(testFolder):
    os.makedirs(testFolder)
if isTrain == 1 or isTest == 1:
    if isTrain == 1:
        SAMPLE_PREFIX = "SAMPLE_PREFIX:O"
        DATASET_NAME = input("Dataset name :")
        DATASET_NAME = "DATASET_NAME:" + DATASET_NAME
        metadataList = combineMetaDataList(DATASET_NAME,
                                           SAMPLES_IN, 
                                           SAMPLE_ROWS, 
                                           SAMPLE_COLS, 
                                           SAMPLE_PREFIX, 
                                           SAMPLE_SUFFIX, 
                                           CLASSES_COUNT, 
                                           BATCHES_COUNT)
        CleanMnistImage(Zero = parameterList[0], One = parameterList[1], Two = parameterList[2], Three = parameterList[3], 
                        Four = parameterList[4], Five = parameterList[5], Six = parameterList[6], Seven = parameterList[7], 
                        Eight = parameterList[8], Nine = parameterList[9], Train = isTrain, Test = 0) 
    if isTest == 1:
        SAMPLE_PREFIX = "SAMPLE_PREFIX:T"
        DATASET_NAME = input("Dataset name :")
        DATASET_NAME = "DATASET_NAME:" + DATASET_NAME
        metadataList = combineMetaDataList(DATASET_NAME,
                                           SAMPLES_IN, 
                                           SAMPLE_ROWS, 
                                           SAMPLE_COLS, 
                                           SAMPLE_PREFIX, 
                                           SAMPLE_SUFFIX, 
                                           CLASSES_COUNT, 
                                           BATCHES_COUNT)
        CleanMnistImage(Zero = parameterList[0], One = parameterList[1], Two = parameterList[2], Three = parameterList[3], 
                        Four = parameterList[4], Five = parameterList[5], Six = parameterList[6], Seven = parameterList[7], 
                        Eight = parameterList[8], Nine = parameterList[9], Train = 0, Test = isTest)
    


