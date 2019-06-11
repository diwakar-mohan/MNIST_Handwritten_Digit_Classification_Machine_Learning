import numpy as np
import pandas as pd
import os
import random
import matplotlib.pyplot as plt
from itertools import combinations
import time
import pickle

import sys


from svmutil import *

def pegasosSGD(inputXMat,labelYMat,wMat,bias,batch_size, threshold,lamda=1.0):
    wCurrMat = np.matrix(np.zeros(wMat.shape,float))        
    I = list(range(inputXMat.shape[0]))
    count = 1
    while True:
        random.shuffle(I)
        atX = inputXMat[I[0:batch_size]]
        atY = labelYMat[I[0:batch_size]]

        J = np.where(np.multiply((atX * wMat.T + bias), atY) < 1.0)
        learning_rate = 1.0 / (lamda * count)
        wCurrMat = (1.0 - learning_rate * lamda) * wMat + (learning_rate * atY[J[0].tolist()].T * atX[J[0].tolist()]) / batch_size
        biasCurr = bias + (learning_rate * np.sum(atY[J[0].tolist()])) / (batch_size * 1.0)

        if abs(wCurrMat - wMat).max() <= threshold and abs(biasCurr - bias).max() <= threshold:
            break
                
        wMat = wCurrMat
        bias = biasCurr
        count += 1
        
    return wCurrMat, biasCurr

def getAccuracy(X_mat,Y_mat,arrClassifier,arrClassPair,outfile):
    is_out_file = False
    if outfile != '':
        out_file_fp = open(outfile,'w')
        is_out_file = True

    val = 0.0
    match = 0.0
    numExamples = 0.0
    accuracy = 0.0
    for i in range(X_mat.shape[0]):
        voting = np.array(np.zeros(10,int))
        X_rowMat = X_mat[i]
        for j in range(arrClassifier.shape[0]):
            w_mat,b_term = arrClassifier[j]
            c1,c2 = arrClassPair[j]
            val = np.sum(w_mat * X_rowMat.T + b_term)
            val_sign = np.sign(val)            
            if val_sign == 1.0 or val_sign == 0.0:
                voting[c1] += 1
            elif val_sign == -1.0:
                voting[c2] += 1

        voting_rev = voting[::-1]
        predict_digit = 9 - (np.argmax(voting_rev))
        if is_out_file:
            out_file_fp.write(str(predict_digit)+'\n')
        else:            
            actual_digit = np.sum(Y_mat[i])
            if predict_digit == actual_digit:
                match += 1
            numExamples += 1
    if is_out_file:
        out_file_fp.close()
    else:
        accuracy = (match * 100.0) / numExamples
    return accuracy            


def drawConfusionMatrix(Y_list_actual, Y_list_predict, numClasses):
    confusion_matrix = np.matrix(np.zeros([numClasses,numClasses]))
    list_size = np.size(Y_list_actual)
    for i in range(list_size):
        confusion_matrix[int(Y_list_predict[i]),int(Y_list_actual[i])] += 1

    return confusion_matrix

def scaleData1(featureMat):
    return (featureMat / 255.0)

def scaleData2(featureMat):
    m = featureMat.shape[0]
    n = featureMat.shape[1]
    scaledMat = np.matrix(np.zeros([m,n], float))
    
    for i in range(featureMat.shape[0]):
        min_ = np.min(featureMat[i])
        max_ = np.max(featureMat[i])
        for j in range(featureMat.shape[1]):
            scaledMat[i,j] = (featureMat[i,j] - min_) / (max_ - min_ )

    return scaledMat;
 
def oneVsOneClassification(X,Y,numClasses,batch_size,epsilon):
    classes = np.arange(numClasses)
    class_examples = np.array(np.zeros(numClasses,int))
    classifier_array = []
    classes_pair = []
    class1 = -1
    class2 = -1

    #store each class examples number
    for i in range(numClasses):
        index = np.where(Y == classes[i])
        class_examples[i] = index[0].shape[0]

    for classifier in combinations(classes,2):
        class1 = classifier[0]
        class2 = classifier[1]

        class1_index = np.where(Y == class1)
        X_class1 = X[class1_index[0].tolist()]    
        Y_class1 = np.matrix(np.ones([class_examples[class1],1]))
        
        class2_index = np.where(Y == class2)
        X_class2 = X[class2_index[0].tolist()]
        Y_class2 = np.matrix([[-1] * 1] * class_examples[class2])
        
        comb_X = np.concatenate((X_class1,X_class2))
        comb_Y = np.concatenate((Y_class1,Y_class2))

        w = np.matrix(np.zeros(X.shape[1],float))
        b = 0.0

        wMatrix, bTerm = pegasosSGD(comb_X,comb_Y,w,b,batch_size,epsilon)
        classifier_array.append((wMatrix,bTerm))
        classes_pair.append((class1,class2))

    classifier_array = np.array(classifier_array)
    classes_pair = np.array(classes_pair)
    return classifier_array,classes_pair

    
#==================================================


nargs = 0
for arg in sys.argv:
    nargs += 1

if nargs == 1:
    #read the dataset from the train file
    trainPath = os.getcwd() + '/train.csv'
    trainData = pd.read_csv(trainPath,header=None,names=None)

    cols = trainData.shape[1]
    X = trainData.iloc[:,0:cols-1]
    X = np.matrix(X.values)
    X = scaleData2(X)

    Y = trainData.iloc[:,cols-1]
    Y = np.matrix(Y.values).T

    #read the dataset from the test file
    testPath = os.getcwd() + '/test.csv'
    testData = pd.read_csv(testPath,header=None,names=None)

    cols = testData.shape[1]
    X_test = testData.iloc[:,0:cols-1]
    X_test = np.matrix(X_test.values)

    X_test = scaleData2(X_test)

    Y_test = testData.iloc[:,cols-1]
    Y_test = np.matrix(Y_test.values).T

    #==================================================
    print("(a): IMPLEMENTING PEGASOS ALGORITHM")
    k = 100             #BATCH SIZE
    W = np.matrix(np.zeros(X.shape[1],float))
    B = 0.0             #BIAS TERM
    numClasses = 10

    epsilon = 0.0003       #301 iters
    print("\n")
    
    #==================================================
    
    print("(b): IMPLEMENTING ONE VS ONE MULTI CLASS SVM")
    CLASSIFIER_ARRAY,CLASSES_PAIR = oneVsOneClassification(X,Y,numClasses,k,epsilon)
    print("\n")

    print("(b): STORING PEGASOS MODEL DATA TO FILE MODEL1")
    model1_file = open("MODEL1","wb")
    MODEL1_DATA = [CLASSIFIER_ARRAY,CLASSES_PAIR]
    pickle.dump(MODEL1_DATA,model1_file)
    model1_file.close()
    print("\n")
  
    #==================================================
    
    print("(b): TRAINING DATA ACCURACY")
    accuracy = getAccuracy(X,Y,CLASSIFIER_ARRAY,CLASSES_PAIR,'')
    print("(b): Training set accuracy = ",accuracy)
    print("\n")

    #==================================================

    print("(b): TEST DATA ACCURACY")
    accuracy = getAccuracy(X_test,Y_test,CLASSIFIER_ARRAY,CLASSES_PAIR,'')
    print("(b): Test set accuracy = ",accuracy)
    print("\n")
    
    #==================================================
    
    print("(c): IMPLEMENTING LIBSVM LINEAR KERNEL")
    linear_model = svm_train(list(Y.flat), X.tolist(), '-s 0 -t 0 -c 1')
    l,a,v = svm_predict(list(Y_test.flat), X_test.tolist(), linear_model)
    print("(c): Linear kernel accuracy = ", a[0])
    print("\n")

    print("(c): STORING LIBSVM LINEAR MODEL DATA TO FILE MODEL2")
    svm_save_model('MODEL2',linear_model)
    print("\n")
    
    print("(c): IMPLEMENTING LIBSVM GAUSSIAN KERNEL")
    gaussian_model1 = svm_train(list(Y.flat), X.tolist(), '-s 0 -t 2 -c 1 -g 0.05')
    l,a,v = svm_predict(list(Y_test.flat), X_test.tolist(), gaussian_model1)
    print("(c): Gaussian kernel accuracy = ", a[0])
    print("\n")

    #==================================================

    print("(d): IMPLEMENTING 10 FOLD CROSS VALIDATION USING LIBSVM GAUSSIAN KERNEL")

    cross_val_acc = []
    test_val_acc = []
    log_c_val = []

    print("(d): LIBSVM CROSS VALIDATION FOR C = 0.00001")
    log_c_val.append(np.log(0.00001))
    accuracy = svm_train(list(Y.flat), X.tolist(), '-s 0 -t 2 -c 0.00001 -g 0.05 -v 10 -h 0')
    print("(d): Cross Validation Accuracy = ", accuracy)
    cross_val_acc.append(accuracy)

    print("(d): LIBSVM TEST ACCURACY FOR C = 0.00001")
    gaussian_model2 = svm_train(list(Y.flat), X.tolist(), '-s 0 -t 2 -c 0.00001 -g 0.05')
    l,a,v = svm_predict(list(Y_test.flat), X_test.tolist(), gaussian_model2)
    print("(d): Test Set Accuracy = ", a[0])
    test_val_acc.append(a[0])
    print("\n")

    print("(d): LIBSVM CROSS VALIDATION FOR C = 0.001")
    log_c_val.append(np.log(0.001))
    accuracy = svm_train(list(Y.flat), X.tolist(), '-s 0 -t 2 -c 0.001 -g 0.05 -v 10 -h 0')
    print("(d): Cross Validation Accuracy = ", accuracy)
    cross_val_acc.append(accuracy)

    print("(d): LIBSVM TEST ACCURACY FOR C = 0.001")
    gaussian_model3 = svm_train(list(Y.flat), X.tolist(), '-s 0 -t 2 -c 0.001 -g 0.05')
    l,a,v = svm_predict(list(Y_test.flat), X_test.tolist(), gaussian_model3)
    print("(d): Test Set Accuracy = ", a[0])
    test_val_acc.append(a[0])
    print("\n")

    print("(d): LIBSVM CROSS VALIDATION FOR C = 1")
    log_c_val.append(np.log(1))
    accuracy = svm_train(list(Y.flat), X.tolist(), '-s 0 -t 2 -c 1 -g 0.05 -v 10 -h 0')
    print("(d): Cross Validation Accuracy = ", accuracy)
    cross_val_acc.append(accuracy)

    print("(d): LIBSVM TEST ACCURACY FOR C = 1")
    gaussian_model4 = svm_train(list(Y.flat), X.tolist(), '-s 0 -t 2 -c 1 -g 0.05')
    l,a,v = svm_predict(list(Y_test.flat), X_test.tolist(), gaussian_model4)
    print("(d): Test Set Accuracy = ", a[0])
    test_val_acc.append(a[0])
    print("\n")

    print("(d): LIBSVM CROSS VALIDATION FOR C = 5")
    log_c_val.append(np.log(5))
    accuracy = svm_train(list(Y.flat), X.tolist(), '-s 0 -t 2 -c 5 -g 0.05 -v 10 -h 0')
    print("(d): Cross Validation Accuracy = ", accuracy)
    cross_val_acc.append(accuracy)

    print("(d): LIBSVM TEST ACCURACY FOR C = 5")
    gaussian_model5 = svm_train(list(Y.flat), X.tolist(), '-s 0 -t 2 -c 5 -g 0.05')
    l,a,v = svm_predict(list(Y_test.flat), X_test.tolist(), gaussian_model5)
    print("(d): Test Set Accuracy = ", a[0])
    test_val_acc.append(a[0])
    print("\n")

    print("(d): LIBSVM CROSS VALIDATION FOR C = 10")
    log_c_val.append(np.log(10))
    accuracy = svm_train(list(Y.flat), X.tolist(), '-s 0 -t 2 -c 10 -g 0.05 -v 10 -h 0')
    print("(d): Cross Validation Accuracy = ", accuracy)
    cross_val_acc.append(accuracy)
    
    print("(d): LIBSVM TEST ACCURACY FOR C = 10")
    gaussian_model6 = svm_train(list(Y.flat), X.tolist(), '-s 0 -t 2 -c 10 -g 0.05')
    l,a,v = svm_predict(list(Y_test.flat), X_test.tolist(), gaussian_model6)
    print("(d): Test Set Accuracy = ", a[0])
    test_val_acc.append(a[0])
    print("\n")

    print("(d): STORING BEST LIBSVM GAUSSIAN MODEL DATA FOR C = 10 TO FILE MODEL3")
    svm_save_model('MODEL3',gaussian_model6)
    print("\n")
    
    print("(e): CONFUSION MATRIX FOR C = 10")
    confusionMat = drawConfusionMatrix(list(Y_test.flat),l,10)
    print("(e): Confusion Matrix = ", confusionMat)
    print("\n")    
    
    print("(d): PLOTTING CROSS VALIDATION ACCURACY AND TEST ACCURACY VS C")
    cross_val_acc = np.array(cross_val_acc)
    test_val_acc = np.array(test_val_acc)
    log_c_val = np.array(log_c_val)

    hFig = plt.figure(figsize=(10,7))
    hPlot = hFig.add_subplot(111)

    hPlot.scatter(log_c_val, cross_val_acc, c='r', s=60, label='10-fold Cross Validation Accuracy')
    hPlot.scatter(log_c_val, test_val_acc, marker='+', c='b', linewidth=2, label='Test Set Accuracy')
    hPlot.set_xlabel('Log C')
    hPlot.set_ylabel('Accuracy')
    hPlot.set_title('LIBSVM Accuracy')
    hPlot.legend()

    plt.show()
    
    #==================================================
else:
    model_ = sys.argv[1]
    input_ = sys.argv[2]
    output_ = sys.argv[3]
    is_svm_model_ = sys.argv[4]

    SVM_MODEL = False
    if is_svm_model_ == "true":
        SVM_MODEL = True

    print("READING INPUT DATA")
    path = os.getcwd() + '/' + input_
    inputData = pd.read_csv(path,header=None,names=None)
    cols_model = inputData.shape[1]
    #assuming file only contains features
    X_model = inputData.iloc[:,0:cols_model-1]
    X_model = np.matrix(X_model.values)
    X_model = scaleData2(X_model)
    Y_model = np.matrix(np.zeros([X_model.shape[0],1]))
    Y_model = Y_model.T
    print("\n")

    print("GENERATING OUTPUT FILE")
    if SVM_MODEL:
        svm_model = svm_load_model(model_)
        l,a,v = svm_predict(list(Y_model.flat),X_model.tolist(),svm_model)
        outfile = open(output_,"w")
        for i in range(len(l)):
            outfile.write(str(l[i])+'\n')
        outfile.close()
    else:
        model_file = open(model_,"rb")
        data = pickle.load(model_file)
        model_file.close()        
        CLASSIFIER_ARRAY_MODEL,CLASSES_PAIR_MODEL = data
        accuracy = getAccuracy(X_model,'',CLASSIFIER_ARRAY_MODEL,CLASSES_PAIR_MODEL,output_)

    print("OUTPUT FILE GENERATED")
