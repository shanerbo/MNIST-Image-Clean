
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-

#GenerateTrainScript


#AddImports


'''
This scripts generate 20 different models for different data sets based on Control variates. 
'''
import math
from pathlib import Path
import numpy as np
import tensorflow as tf
import pandas as pd

def output(filePath, labelPath, n1,n2,n3,n4,n5,n6,imageSize,batchSize):
    


    #*********************************************************************************
    #AddConsts

    ImgWidth = 100
    ImgHeight = 100
    FirstImageIndex = 1
    LastImageIndex = imageSize
    L1_Neurons =n1
    L2_Neurons =n2
    L3_Neurons =n3
    L4_Neurons =n4
    L5_Neurons =n5
    L6_Neurons =n6
    numcorrect = 0
    totalnum = 0
    HowManyBatches = batchSize
    NumClasses =2
    NumOfImages = LastImageIndex - FirstImageIndex + 1
    SamplesPerBatch = int(NumOfImages/HowManyBatches)



    #*********************************************************************************
    #AddOpenImageData

    from PIL import Image
    ImageCount = 0
    OneImagePixelSize = ImgWidth * ImgHeight
    AllImagesData = np.zeros((NumOfImages, ImgWidth, ImgHeight, 1), dtype=float)

    for ImagesIndex in range(FirstImageIndex, LastImageIndex + 1):
        FilePath = Path(filePath + str(ImagesIndex) +".png")
        imP0 = Image.open(FilePath)
        pixP0 = np.asarray(imP0)
        #Returns dimension of image
        imageShape = pixP0.shape
        numDim = len(imageShape)
        for row_index in range(0, ImgHeight):
            for col_index in range(0, ImgWidth):
                all3Channels = pixP0[row_index, col_index]
                if(numDim == 2):
                    AllImagesData[ImageCount][row_index][col_index] = all3Channels
                elif(numDim ==3):
                    AllImagesData[ImageCount][row_index][col_index] = all3Channels[0]
                else:
                    print("Unknown number of dimensions for input")

        ImageCount = ImageCount + 1

    AllImagesData = AllImagesData.reshape(NumOfImages, ImgHeight, ImgWidth)


    #*********************************************************************************
    #AddLoadLabels


    LabelsPath = Path(labelPath)
    f = open(LabelsPath, 'r')
    x = f.read()
    mylines = x.split(',')
    f.close()
    label_size = len(mylines)
    classes = 2
    AllImagesLabels = np.zeros(shape = (label_size, classes))
    for LabelIndex in range(0,label_size):
        WhichHot = (int)(mylines[LabelIndex])
        AllImagesLabels[LabelIndex][WhichHot] = 1


    #*********************************************************************************
    #AddConvolutionLogic


    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, [None, ImgWidth, ImgHeight, 1],  name = "X_SAVE")
    Y_CorrectOneBatch_PH = tf.placeholder(tf.float32, [None, NumClasses],  name = "Y_CorrectOneBatch_PH_SAVE")

    W1 = tf.Variable(tf.truncated_normal([5, 5, 1, L1_Neurons], stddev=0.1), name="W1_SAVE")
    B1 = tf.Variable(tf.ones([L1_Neurons])/NumClasses, name="B1_SAVE")
    W2 = tf.Variable(tf.truncated_normal([5, 5, L1_Neurons, L2_Neurons], stddev=0.1), name="W2_SAVE")
    B2 = tf.Variable(tf.ones([L2_Neurons])/NumClasses, name="B2_SAVE")
    W3 = tf.Variable(tf.truncated_normal([4, 4, L2_Neurons, L3_Neurons], stddev=0.1), name="W3_SAVE")
    B3 = tf.Variable(tf.ones([L3_Neurons])/NumClasses, name="B3_SAVE")
    W4 = tf.Variable(tf.truncated_normal([4, 4, L3_Neurons, L4_Neurons], stddev=0.1), name="W4_SAVE")
    B4 = tf.Variable(tf.ones([L4_Neurons])/NumClasses, name="B4_SAVE")
    W5 = tf.Variable(tf.truncated_normal([ 13 * 13 * L4_Neurons, L5_Neurons], stddev=0.1), name="W5_SAVE")
    B5 = tf.Variable(tf.ones([L5_Neurons])/NumClasses, name="B5_SAVE")
    W6 = tf.Variable(tf.truncated_normal([L5_Neurons, L6_Neurons], stddev=0.1), name="W6_SAVE")
    B6 = tf.Variable(tf.ones([L6_Neurons])/NumClasses, name="B6_SAVE")

    stride = 1
    Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
    stride = 2
    Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
    Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)
    Y4 = tf.nn.relu(tf.nn.conv2d(Y3, W4, strides=[1, stride, stride, 1], padding='SAME') + B4)
    Y4_RESHAPE = tf.reshape(Y4, shape=[-1,13 * 13 * L4_Neurons])
    Y5 = tf.nn.relu(tf.matmul(Y4_RESHAPE, W5) + B5)
    Ylogits = tf.matmul(Y5, W6) + B6

    MySoftmaxPrediction = tf.nn.softmax(Ylogits, name = "MySoftmaxPredictionOperation")

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_CorrectOneBatch_PH)
    loss = tf.multiply(tf.reduce_mean(cross_entropy), 100, name = "LossOperation")
    step = tf.placeholder(tf.int32)
    lr = tf.placeholder(tf.float32)
    train_step = tf.train.AdamOptimizer(lr ).minimize(loss)

    myArgMaxPrediction = tf.argmax(MySoftmaxPrediction, axis=1, name="ArgMaxPredictionOperation")
    myArgMaxCorrect =  tf.argmax(Y_CorrectOneBatch_PH, axis=1, name="ArgMaxCorrectOperation")


    #*********************************************************************************
    #AddRun

    sess =  tf.Session()
    batch_X = np.zeros(shape = (SamplesPerBatch, ImgWidth, ImgHeight, 1))
    batch_Y = np.zeros(shape = (SamplesPerBatch, NumClasses))
    Init = tf.global_variables_initializer()
    sess.run(Init)
    AllImagesIndex = 0
    max_learning_rate = 0.003
    min_learning_rate = 0.0001
    decay_speed = 2000.0
    BatchIndex = 0
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-BatchIndex/decay_speed)
    train_data = {X: batch_X, Y_CorrectOneBatch_PH: batch_Y, step: BatchIndex, lr:learning_rate }


    for BatchIndex in range(0, HowManyBatches):

        for SampleIndex in range(0, SamplesPerBatch):

            for hotindex in range(0, NumClasses):
                batch_Y[SampleIndex][hotindex] = AllImagesLabels[AllImagesIndex][hotindex]

            for Rowindex in range(0, ImgHeight):
                for Colindex in range(0, ImgWidth):
                    batch_X[SampleIndex][Rowindex][Colindex][0]  =    AllImagesData[AllImagesIndex][Rowindex][Colindex]

            AllImagesIndex = AllImagesIndex + 1


        outs = sess.run(loss, feed_dict = train_data)
        batch_X_3D = batch_X.reshape(SamplesPerBatch, ImgHeight, ImgWidth)
        sess.run(train_step, feed_dict = train_data)


        RunPredictions = sess.run(myArgMaxPrediction, feed_dict = train_data)
        RunCorrects = sess.run(myArgMaxCorrect, feed_dict = train_data)


        for SampleIndex in range(0, SamplesPerBatch):
            if(RunPredictions[SampleIndex] == RunCorrects[SampleIndex]):
                numcorrect = numcorrect + 1


    return numcorrect


# In[2]:



# In[3]:


FilePath = ["D:\\TFS\\Research\\SR71\\data\\img\\png\\Cell_1\\TRAIN\\O",
            "D:\\TFS\\Research\\SR71\\data\\img\\png\\Cell_1\\HANDWRITE\\O",
            "D:\\TFS\\Research\\SR71\\data\\img\\png\\Cell_1\\HANDWRITE_NEW\\O",
            "D:\\TFS\\Research\\SR71\\data\\img\\png\\Cell_1\\LARGEDATA\\O"
           ]
		   # There are four different data sets
LabelPath = ["D:\\TFS\\Research\\SR71\\data\\img\\png\\Cell_1\\TRAIN\\labels.txt",
             "D:\\TFS\\Research\\SR71\\data\\img\\png\\Cell_1\\HANDWRITE\\labels.txt",
             "D:\\TFS\\Research\\SR71\\data\\img\\png\\Cell_1\\HANDWRITE_NEW\\labels.txt",
             "D:\\TFS\\Research\\SR71\\data\\img\\png\\Cell_1\\LARGEDATA\\labels.txt"
            ]
			# For each data set, it has its own labels.txt file


# In[4]:


repeatTime = [50]
# For each data set, it will be run 50 times to get an average so that it can avoid bias.

# In[14]:


import numpy as np
image_size = [60, 160, 2560, 12060]
# This list store size of images of each data set
result_60i_2_4_16_32_8 = []
result_60i_4_4_16_16_32 = []
result_60i_4_8_16_32_32 = []
result_60i_8_8_32_64_16 = []
result_60i_16_32_48_64_32 = []
for i in range(repeatTime[0]):
    result_60i_2_4_16_32_8.append(output(FilePath[0], LabelPath[0],2,4,16,32,8,2,image_size[0],30)/image_size[0])
    result_60i_4_4_16_16_32.append(output(FilePath[0], LabelPath[0],4,4,16,16,32,2,image_size[0],30)/image_size[0])
    result_60i_4_8_16_32_32.append(output(FilePath[0], LabelPath[0],4,8,16,32,32,2,image_size[0],30)/image_size[0])
    result_60i_8_8_32_64_16.append(output(FilePath[0], LabelPath[0],8,8,32,64,16,2,image_size[0],30)/image_size[0])
    result_60i_16_32_48_64_32.append(output(FilePath[0], LabelPath[0],16,32,48,64,32,2,image_size[0],30)/image_size[0])


# In[6]:


result_160i_2_4_16_32_8 = []
result_160i_4_4_16_16_32 = []
result_160i_4_8_16_32_32 = []
result_160i_8_8_32_64_16 = []
result_160i_16_32_48_64_32 = []
for i in range(repeatTime[0]):
    result_160i_2_4_16_32_8.append(output(FilePath[1], LabelPath[1],2,4,16,32,8,2,image_size[1],20)/image_size[1])
    result_160i_4_4_16_16_32.append(output(FilePath[1], LabelPath[1],4,4,16,16,32,2,image_size[1],20)/image_size[1])
    result_160i_4_8_16_32_32.append(output(FilePath[1], LabelPath[1],4,8,16,32,32,2,image_size[1],20)/image_size[1])
    result_160i_8_8_32_64_16.append(output(FilePath[1], LabelPath[1],8,8,32,64,16,2,image_size[1],20)/image_size[1])
    result_160i_16_32_48_64_32.append(output(FilePath[1], LabelPath[1],16,32,48,64,32,2,image_size[1],20)/image_size[1])


# In[7]:


result_2560i_2_4_16_32_8 = []
result_2560i_4_4_16_16_32 = []
result_2560i_4_8_16_32_32 = []
result_2560i_8_8_32_64_16 = []
result_2560i_16_32_48_64_32 = []
for i in range(repeatTime[0]):
    result_2560i_2_4_16_32_8.append(output(FilePath[2], LabelPath[2],2,4,16,32,8,2,image_size[2],6)/image_size[2])
    result_2560i_4_4_16_16_32.append(output(FilePath[2], LabelPath[2],4,4,16,16,32,2,image_size[2],6)/image_size[2])
    result_2560i_4_8_16_32_32.append(output(FilePath[2], LabelPath[2],4,8,16,32,32,2,image_size[2],6)/image_size[2])
    result_2560i_8_8_32_64_16.append(output(FilePath[2], LabelPath[2],8,8,32,64,16,2,image_size[2],6)/image_size[2])
    result_2560i_16_32_48_64_32.append(output(FilePath[2], LabelPath[2],16,32,48,64,32,2,image_size[2],6)/image_size[2])


# In[ ]:


result_12060i_2_4_16_32_8 = []
result_12060i_4_4_16_16_32 = []
result_12060i_4_8_16_32_32 = []
result_12060i_8_8_32_64_16 = []
result_12060i_16_32_48_64_32 = []
for i in range(repeatTime[0]):
    result_12060i_2_4_16_32_8.append(output(FilePath[3], LabelPath[3],2,4,16,32,8,2,image_size[3],30)/image_size[3])
    result_12060i_2_4_16_32_8.append(output(FilePath[3], LabelPath[3],4,4,16,16,32,2,image_size[3],30)/image_size[3])
    result_12060i_2_4_16_32_8.append(output(FilePath[3], LabelPath[3],4,8,16,32,32,2,image_size[3],30)/image_size[3])
    result_12060i_2_4_16_32_8.append(output(FilePath[3], LabelPath[3],8,8,32,64,16,2,image_size[3],30)/image_size[3])
    result_12060i_2_4_16_32_8.append(output(FilePath[3], LabelPath[3],16,32,48,64,32,2,image_size[3],30)/image_size[3])


# In[47]:


d = {'result_60i_2_4_16_32_8': result_60i_2_4_16_32_8, 
     'result_60i_4_4_16_16_32': result_60i_4_4_16_16_32,
     'result_60i_4_8_16_32_32': result_60i_4_8_16_32_32,
     'result_60i_8_8_32_64_16': result_60i_8_8_32_64_16,
     'result_60i_16_32_48_64_32': result_60i_16_32_48_64_32,
     'result_160i_2_4_16_32_8': result_160i_2_4_16_32_8,
     'result_160i_4_4_16_16_32': result_160i_4_4_16_16_32,
     'result_160i_4_8_16_32_32': result_160i_4_8_16_32_32,
     'result_160i_8_8_32_64_16': result_160i_8_8_32_64_16,
     'result_160i_16_32_48_64_32': result_160i_16_32_48_64_32,
     'result_2560i_2_4_16_32_8': result_2560i_2_4_16_32_8,
     'result_2560i_4_4_16_16_32': result_2560i_4_4_16_16_32,
     'result_2560i_4_8_16_32_32': result_2560i_4_8_16_32_32,
     'result_2560i_8_8_32_64_16': result_2560i_8_8_32_64_16,
     'result_2560i_16_32_48_64_32': result_2560i_16_32_48_64_32,
     'result_12060i_2_4_16_32_8': result_12060i_2_4_16_32_8,
     'result_12060i_4_4_16_16_32': result_12060i_4_4_16_16_32,
     'result_12060i_4_8_16_32_32': result_12060i_4_8_16_32_32,
     'result_12060i_8_8_32_64_16': result_12060i_8_8_32_64_16,
     'result_12060i_16_32_48_64_32': result_12060i_16_32_48_64_32}


# In[48]:


df = pd.DataFrame(data=d)


# In[51]:


df1 = df[['result_60i_2_4_16_32_8',
          'result_60i_4_4_16_16_32',
          'result_60i_4_8_16_32_32',
          'result_60i_8_8_32_64_16',
          'result_60i_16_32_48_64_32',
          'result_160i_2_4_16_32_8',
          'result_160i_4_4_16_16_32',
          'result_160i_4_8_16_32_32',
          'result_160i_8_8_32_64_16',
          'result_160i_16_32_48_64_32',
          'result_2560i_2_4_16_32_8',
          'result_2560i_4_4_16_16_32',
          'result_2560i_4_8_16_32_32',
          'result_2560i_8_8_32_64_16',
          'result_2560i_16_32_48_64_32',
          'result_12060i_2_4_16_32_8',
          'result_12060i_4_4_16_16_32',
          'result_12060i_4_8_16_32_32',
          'result_12060i_8_8_32_64_16',
          'result_12060i_16_32_48_64_32']]


# In[55]:


df1.to_csv("CHC_AI_LAB_BENCHMARK.csv", index = False, sep=',', encoding='utf-8')
# save numpy n-dimension array into csv file.

