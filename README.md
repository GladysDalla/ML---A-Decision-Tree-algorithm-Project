# ML---A-Decision-Tree-algorithm-Project

In this assignment you will use scikit-learn (sklearn) library to evaluate decision trees.

Setup:

1. Please go through sklearn.pptx slides.  If needed, go over IDLE.pptx slides.

2. Choose any two datasets from OpenML (https://www.openml.org/search?type=data). Each must have:

    Binary nominal target (i.e. binary classification task)
        Use "Filter results" button on the top right corner of the OpenML web page and select Target -> Binary classification
    All numeric features (sklearn decision trees do not directly handle nominal features)
        Use "Sort results" button on the top right corner and select Numeric Features (top ones will be all numeric features; verify it)
    At least 1000 examples
        Use "Filter results" and select Instances -> 1000s

Assignment Task:

Separately on each dataset, compare decision trees created using the entropy and gini index criteria by plotting ROC curves and computing the AUC values. Generate results using 10-fold cross-validation. Do parameter search on min_samples_leaf parameter using GridsearchCV  (at least 5 different values). Write a program to do the entire task (i.e. do not do it from the Python prompt as in the slides).


PROJECT

A decision tree algorithm project
A brief description for each dataset (what is the task, what are the features and the target)
Dataset 1: ID: 1462 banknote-authentication
URL; https://www.openml.org/search?type=data&sort=runs&status=active&qualities.NumberOfClasses=%3D_2&id=1462
Data were extracted from images that were taken from genuine and forged banknote-like specimens. For digitization, an industrial camera usually used for print inspection was used. The final images have 400x 400 pixels. Due to the object lens and distance to the investigated object gray-scale pictures with a resolution of about 660 dpi were gained. A Wavelet Transform tool was used to extract features from these images. (Source: https://www.openml.org/d/1462).
Task: This dataset is about distinguishing genuine and forged banknotes. The task is to model the probability that a banknote is fraudulent as its features function. The number of instances (rows) in the data set is 1372, and the number of features (columns) is 5. 

Dataset 2: ID: 40983 (Wilt Data Set)
URL; https://www.openml.org/search?type=data&sort=runs&status=active&qualities.NumberOfClasses=%3D_2&id=40983
This data set involved detecting diseased trees in Quickbird satellite images. The high-resolution QuickBird images of 165 different area were acquired in 27 August 2012 to detect the diseased trees. The QuickBird images contains four 2.4 m resolution MS bands; G, R, NIR and PAN band. 
Task: The dataset is about the detection of Pine Wilt Disease (PWD) infected trees and non-affected. The number of instances (rows) in the data set is 4839, and the number of features (columns) is 6.
