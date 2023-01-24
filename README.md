# Linear_regression_project

Need to update: I standardized data before splitting but that is wrongly using future data.

You first need to split the data into training and test set (validation set could be useful too).

Don't forget that testing data points represent real-world data. Feature normalization (or data standardization) of the explanatory (or predictor) variables is a technique used to center and normalise the data by subtracting the mean and dividing by the variance. If you take the mean and variance of the whole dataset you'll be introducing future information into the training explanatory variables (i.e. the mean and variance).

Therefore, you should perform feature normalisation over the training data. Then perform normalisation on testing instances as well, but this time using the mean and variance of training explanatory variables. In this way, we can test and evaluate whether our model can generalize well to new, unseen data points.

For a more comprehensive read, you can read my article Feature Scaling and Normalisation in a nutshell

As an example, assuming we have the following data:

>>> import numpy as np
>>> 
>>> X, y = np.arange(10).reshape((5, 2)), range(5)
where X represents our features:

>>> X
[[0 1]
 [2 3]
 [4 5]
 [6 7]
 [8 9]]
and Y contains the corresponding label

>>> list(y)
>>> [0, 1, 2, 3, 4]
Step 1: Create training/testing sets

>>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

>>> X_train
[[4 5]
 [0 1]
 [6 7]]
>>>
>>> X_test
[[2 3]
 [8 9]]
>>>
>>> y_train
[2, 0, 3]
>>>
>>> y_test
[1, 4]
Step 2: Normalise training data

>>> from sklearn import preprocessing
>>> 
>>> normalizer = preprocessing.Normalizer()
>>> normalized_train_X = normalizer.fit_transform(X_train)
>>> normalized_train_X
array([[0.62469505, 0.78086881],
       [0.        , 1.        ],
       [0.65079137, 0.7592566 ]])
Step 3: Normalize testing data

>>> normalized_test_X = normalizer.transform(X_test)
>>> normalized_test_X
array([[0.5547002 , 0.83205029],
       [0.66436384, 0.74740932]])
