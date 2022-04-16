# KNN
Code KNN from scratch to determine accuracy of MEDV price for Boston Housing Dataset. Here I perform descriptive analysis use visualizations like histograms and scatter plot between different features to analyse and comprehend relevance of the features to target feature. 
# Part 1: Descriptive analysis
## 1.1: Histogram of housing prices
## 1.2: Scatter plot of housing prices and crime
# Part 2: Experimental Setup
## 2.1 Begin by writing a function to compute the Root Mean Squared Error for a list of numbers

"""
Function
--------
compute_rmse

Given two arrays, one of actual values and one of predicted values,
compute the Roote Mean Squared Error

Parameters
----------
predictions : array
    Array of numerical values corresponding to predictions for each of the N observations

yvalues : array
    Array of numerical values corresponding to the actual values for each of the N observations

Returns
-------
rmse : int
    Root Mean Squared Error of the prediction

Example
-------
>>> print compute_rmse((2,2,3),(0,2,6)
2.16
"""

## 2.2 Divide your data into training and testing datasets

## 2.3 Use a very bad baseline for prediction, and compute RMSE

# Part 3: Nearest Neighbors
## 3.1 Nearest Neighbors: Distance function

"""
Function
--------
distance

Given two instances and a value for L, return the L-Norm distance between them

Parameters
----------
x1, x2 : array
    Array of numerical values corresponding to predictions for each of the N observations

L: int
    Value of L to use in computing distances

Returns
-------
dist : int
    The L-norm distance between instances

Example
-------
>>> print distance((3,4),(6,8),2)
7

"""
## 3.2 Basic Nearest Neighbor algorithm
"""
Function
--------
nneighbor

Given the training data set, testing data set, the target variable values of training data set, 
K = 1 (since this is nearest neighbor) and L = 2 (for Euclidean Distance), predict the target variable for 
testing data.

K = 1 set's just 1 data point as a reference data point, we can later on re-use this as K-NN where k != 1

Parameters
----------
x_train (bdata_train), y, x_input (bdata_test) : array
    Array of numerical values corresponding to CRIM and RM features for each of the training observations
k:int
    Determines the count of points reference to which the distances will be calculated
L: int
    Value of L to use in computing distances

Returns
-------
pred : int
    The predicted value of target variable corresponding to each test set entry (i.e. bdata_test)

Example
-------
>>> [5.5,6.5]...(train entries), [24.2], [4.9,6.7]...(test entries), 1, 2)

18.9

"""
## 3.3 Results and Normalization
"""
Function
--------
normalize

Given the data, convert the integer into z-scale 

Parameters
----------
raw_data: array
    Numerical array

Returns
-------
normalized_data : array
    Numeric array within -1 to 1 range

Example
-------
>>> 0.14932  5.741
.
.
.

-0.383740 -0.812161
.
.
.

"""
## 3.4 Optimization
## 3.5 Cross-Validation

"""
Function
--------
cross_validation_split

Given the dataset and number of folds, split the dataset into those many number of folds

Parameters
----------
dataset: dataframe

folds: integer

Returns
-------
dataset_split : list
    number of indexes in list = number of folds

Example
-------
>>> [0.14932  5.741 22.4 ] [5.14932  5.741 22.4 ] [9.14932  5.741 22.4 ] [3.14932  5.741 22.4 ] [2.14932  5.741 22.4 ]
[9.14932  5.741 22.4 ] [1.14932  5.741 22.4 ] [10.14932  5.741 22.4 ] [8.14932  5.741 22.4 ] [6.14932  5.741 22.4 ]
, 5
.
.
.

list[0] --> [0.14932  5.741 22.4 ]
            [9.14932  5.741 22.4 ]
list[1] --> [6.14932  5.741 22.4 ]
            [8.14932  5.741 22.4 ]
list[2] --> [1.14932  5.741 22.4 ]
            [2.14932  5.741 22.4 ]
list[3] --> [5.14932  5.741 22.4 ]
            [9.14932  5.741 22.4 ]
list[4] --> [10.14932  5.741 22.4 ]
            [3.14932  5.741 22.4 ]
"""
## 3.6 K-fold CV

"""
Function
--------
kfoldCV

Given the dataset, number of folds, value of k for k-NN and number of features in the dataset:
1) Predict the value of target variable for each fold (if 10 folds, 1 used as test and other 9 are used as train
in 1st iteration and this continues for 10 iterations where the test data changes in each iteration)
2) Return RMSE array

Parameters
----------
dataset: dataframe

f: integer

K: K value (=1 for NN) and any other value for K-NN

no_of_features = this variable solely handles the nymber of features to be used to predict a value from training set

Returns
-------
result : list of size folds
    Numeric value

Example
-------

"""
## 3.7 3.6 K-Nearest Neighbors Algorithm
