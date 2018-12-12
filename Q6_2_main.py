'''
README
 
This sit he main file which calls all other funcitons like likelihood and naiveBayesClassify.
If you run this file, you see the output for all the functions

'''

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from collections import Counter
import scipy.io
from prior import *
from likelihood import *
from naiveBayesClassify import *


mat = scipy.io.loadmat('ecoli.mat')

xTrain=mat['xTrain']
yTrain=mat['yTrain']
xTest=mat['xTest']
yTest=mat['yTest']

def convert_to_list(mat_obj):
    y=[]
    for i in mat_obj:
        y.append(i[0])
    return y

yTrain=convert_to_list(yTrain)
#xTrain=convert_to_list(xTrain)
#xTest=convert_to_list(xTest)
yTest=convert_to_list(yTest)

#print(xTrain[0:10],yTrain[:10])

p=prior(yTrain)
print('Prior:',p)

M,V=likelihood(xTrain,yTrain)
print("Likelihood: ", M)
print("Variance: ", V)


nb=(naiveBayesClassify(xTest, M, V, p))
print("Predicted classes: ",nb)

##Analysis of predictions
#fraction classified correctly
match_count=0
class_1_pred=0
class_2_pred=0
class_3_pred=0
class_4_pred=0
class_5_pred=0

class1_trpos=0
class2_trpos=0
class3_trpos=0
class4_trpos=0
class5_trpos=0

precision_array=[]
recall_array=[]

for i in range(len(nb)):
    if nb[i]==yTest[i]:
        match_count+=1
    if nb[i]==1:
        class_1_pred+=1
        if yTest[i]==1:
            class1_trpos+=1
    if nb[i]==2:
        class_2_pred+=1
        if yTest[i]==2:
            class2_trpos+=1
    if nb[i]==3:
        class_3_pred+=1
        if yTest[i]==3:
            class3_trpos+=1
    if nb[i]==4:
        class_4_pred+=1
        if yTest[i]==4:
            class4_trpos+=1
    if nb[i]==5:
        class_5_pred+=1
        if yTest[i]==5:
            class5_trpos+=1
        

print("Fraction of test samples classified correctly: ",float(match_count)/float(len(nb))*100)

#precision of class 1
class1_prec=float(class1_trpos)/float(class_1_pred)
print("Precision of class 1: ",class1_prec)

#recall
i=0
for x in yTest:
    if x==1:
        i+=1

recall=float(class1_trpos)/float(i)
print("Recall of class 1: ",recall)

#precision of class 5
class5_prec=float(class5_trpos)/float(class_5_pred)
print("Precision of class 5: ",class5_prec)

#recall
i=0
for x in yTest:
    if x==5:
        i+=1

recall=float(class5_trpos)/float(i)
print("Recall of class 5: ",recall)
