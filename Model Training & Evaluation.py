
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.2** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-machine-learning/resources/bANLa) course resource._
# 
# ---

# # Assignment 3 - Evaluation
# 
# In this assignment you will train several models and evaluate how effectively they predict instances of fraud using data based on [this dataset from Kaggle](https://www.kaggle.com/dalpozz/creditcardfraud).
#  
# Each row in `fraud_data.csv` corresponds to a credit card transaction. Features include confidential variables `V1` through `V28` as well as `Amount` which is the amount of the transaction. 
#  
# The target is stored in the `class` column, where a value of 1 corresponds to an instance of fraud and 0 corresponds to an instance of not fraud.

# In[1]:

import numpy as np
import pandas as pd


# ### Question 1
# Import the data from `fraud_data.csv`. What percentage of the observations in the dataset are instances of fraud?
# 
# *This function should return a float between 0 and 1.* 

# In[2]:

def answer_one():
    
    df_fraud = pd.read_csv('./fraud_data.csv')
    y = df_fraud['Class']
    
    return np.mean(y)


# In[3]:

# Use X_train, X_test, y_train, y_test for all of the following questions
from sklearn.model_selection import train_test_split

df = pd.read_csv('./fraud_data.csv')

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# ### Question 2
# 
# Using `X_train`, `X_test`, `y_train`, and `y_test` (as defined above), train a dummy classifier that classifies everything as the majority class of the training data. What is the accuracy of this classifier? What is the recall?
# 
# *This function should a return a tuple with two floats, i.e. `(accuracy score, recall score)`.*

# In[4]:

def answer_two():
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import recall_score, accuracy_score
    
    dummy = DummyClassifier(strategy = 'most_frequent')
    dummy.fit(X_train,y_train)
    
    accuracy = dummy.score(X_test,y_test)
    recall = recall_score(y_test,dummy.predict(X_test))
    
    return (accuracy, recall)


# ### Question 3
# 
# Using X_train, X_test, y_train, y_test (as defined above), train a SVC classifer using the default parameters. What is the accuracy, recall, and precision of this classifier?
# 
# *This function should a return a tuple with three floats, i.e. `(accuracy score, recall score, precision score)`.*

# In[5]:

def answer_three():
    from sklearn.metrics import recall_score, precision_score
    from sklearn.svm import SVC

    svm = SVC()
    svm.fit(X_train,y_train)
    y_pred = svm.predict(X_test)
    
    accuracy = svm.score(X_test,y_test)
    recall = recall_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    
    return (accuracy, recall, precision)


# ### Question 4
# 
# Using the SVC classifier with parameters `{'C': 1e9, 'gamma': 1e-07}`, what is the confusion matrix when using a threshold of -220 on the decision function. Use X_test and y_test.
# 
# *This function should return a confusion matrix, a 2x2 numpy array with 4 integers.*

# In[10]:

def answer_four():
    from sklearn.metrics import confusion_matrix
    from sklearn.svm import SVC
    
    clf_svc = SVC(C = 1e9, gamma = 1e-07).fit(X_train,y_train)
    y_score_svc = clf_svc.decision_function(X_test)
    y_pred = y_score_svc > -220
    cm = confusion_matrix(y_test,y_pred)
    
    return cm


# ### Question 5
# 
# Train a logisitic regression classifier with default parameters using X_train and y_train.
# 
# For the logisitic regression classifier, create a precision recall curve and a roc curve using y_test and the probability estimates for X_test (probability it is fraud).
# 
# Looking at the precision recall curve, what is the recall when the precision is `0.75`?
# 
# Looking at the roc curve, what is the true positive rate when the false positive rate is `0.16`?
# 
# *This function should return a tuple with two floats, i.e. `(recall, true positive rate)`.*

# In[11]:

def answer_five():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_recall_curve, roc_curve
    
    LR = LogisticRegression()
    LR.fit(X_train,y_train)
    
    y_pred = LR.predict_proba(X_test)
    precision, recall, threshold_pr = precision_recall_curve(y_test, y_pred[:,1])
    fpr, tpr, threshold_roc = roc_curve(y_test, y_pred[:,1])
    
    
    recall = recall[np.argmin(abs(precision - 0.75))]
    true_positive_rate = tpr[np.argmin(abs(fpr - 0.16))]
    
    return (recall, true_positive_rate)


# ### Question 6
# 
# Perform a grid search over the parameters listed below for a Logisitic Regression classifier, using recall for scoring and the default 3-fold cross validation.
# 
# `'penalty': ['l1', 'l2']`
# 
# `'C':[0.01, 0.1, 1, 10, 100]`
# 
# From `.cv_results_`, create an array of the mean test scores of each parameter combination. i.e.
# 
# |      	| `l1` 	| `l2` 	|
# |:----:	|----	|----	|
# | **`0.01`** 	|    ?	|   ? 	|
# | **`0.1`**  	|    ?	|   ? 	|
# | **`1`**    	|    ?	|   ? 	|
# | **`10`**   	|    ?	|   ? 	|
# | **`100`**   	|    ?	|   ? 	|
# 
# <br>
# 
# *This function should return a 5 by 2 numpy array with 10 floats.* 
# 
# *Note: do not return a DataFrame, just the values denoted by '?' above in a numpy array. You might need to reshape your raw result to meet the format we are looking for.*

# In[12]:

def answer_six():    
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression

    params = {'penalty':['l1','l2'],'C':[0.01, 0.1, 1, 10, 100]}
    
    LR = LogisticRegression()
    grid = GridSearchCV(estimator=LR, param_grid=params,cv=3,scoring = 'recall')
    grid.fit(X_train,y_train)
    
    scores = np.array(grid.cv_results_['mean_test_score']).reshape(-1,2) 
    
    return scores


# In[13]:

# Use the following function to help visualize results from the grid search
def GridSearch_Heatmap(scores):
    get_ipython().magic('matplotlib inline')
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure()
    sns.heatmap(scores.reshape(5,2), xticklabels=['l1','l2'], yticklabels=[0.01, 0.1, 1, 10, 100])
    plt.yticks(rotation=0);

# GridSearch_Heatmap(answer_six())


# In[ ]:



