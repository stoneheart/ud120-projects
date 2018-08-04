#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

#########################################################
# clf = SVC(kernel='linear')
clf = SVC(kernel='rbf', C=10000.0)

# features_train = features_train[:len(features_train)/100] 
# labels_train = labels_train[:len(labels_train)/100]

# train the classifier
t0 = time()
clf.fit(features_train, labels_train)
print "training time: %.3f s" % (time()-t0)

# make predictions
t1 = time()
predict = clf.predict(features_test)
print "predicting time: %.3f s" % (time()-t1)

# get the accuracy
accuracy = clf.score(features_test, labels_test)
print accuracy

# print preditoins
print predict[10]
print predict[26]
print predict[50]
print sum(predict)
