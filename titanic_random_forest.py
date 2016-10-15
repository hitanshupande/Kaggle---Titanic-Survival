# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 00:46:08 2016

@author: Hitanshu
"""

from sklearn.ensemble import RandomForestRegressor
#Using the ROC/AUC error metric
from sklearn.metrics import roc_auc_score
import pandas as pd

x = pd.read_csv("train.csv")
y = x.pop("Survived")
test = pd.read_csv("test.csv")


#Build a first model using only non-categorical data
x["Age"].fillna(x.Age.mean(), inplace=True)

x.describe()

#Build a dataset with only numerical data
numeric_variables = list(x.dtypes[x.dtypes != object].index)
x[numeric_variables].head()


#Build our first model using RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=42)
model.fit(x[numeric_variables], y)

#For regression, the oob_score_ attribute gives the R^2 based on the oob predictions. 
model.oob_score_

y_oob = model.oob_prediction_
print("c-stat: ", roc_auc_score(y, y_oob))

#Using this as the benchmark score, we can go forward and include categorical variables in our train dataset

#Let us do some EDA on our variables
def describe_categorical(x):
    #similar to describe() function, but for categorical values
    from IPython.display import display, HTML
    display(HTML(x[x.columns[x.dtypes == "object"]].describe().to_html()))

describe_categorical(x)


#Dropping some categorical variables for now
x.drop(["Name", "Ticket"], axis=1, inplace=True)

#Change cabin variable so it only contains first letter or None
def clean_cabin(x):
    try:
        return x[0]
    except TypeError:
        return "None"
            
x["Cabin"] = x.Cabin.apply(clean_cabin)


categorical_variables = ['Sex', 'Cabin', 'Embarked']

for variable in categorical_variables:
    #Fill missing data with the word 'Missing'
    x[variable].fillna("Missing", inplace=True)
    #Create array of dummies
    dummies = pd.get_dummies(x[variable], prefix=variable)
    #Update x to include dummies and drop the main variable
    x = pd.concat([x, dummies], axis=1)
    x.drop([variable], axis=1, inplace=True)
    
x.head()


#Building the model
model = RandomForestRegressor(100, oob_score=True, n_jobs=-1, random_state=42)
model.fit(x,y)
print("C-stat: ", roc_auc_score(y, model.oob_prediction_))



#Tuning various model parameters

results = []
n_estimator_options = [30, 50, 100, 200, 500, 1000, 2000]

for trees in n_estimator_options:
    model = RandomForestRegressor(trees, oob_score=True, n_jobs=-1, random_state=42)
    model.fit(x,y)
    print(trees, "trees")
    roc = roc_auc_score(y, model.oob_prediction_)
    print("C-stat: ", roc)
    results.append(roc)
    print ("")
    
pd.Series(results, n_estimator_options).plot();

results = []
max_features_options = ["auto", None, "sqrt", "log2", 0.9, 0.2]

for max_features in max_features_options:
    model = RandomForestRegressor(n_estimators=1000, oob_score=True, n_jobs=-1, random_state=42, max_features=max_features)
    model.fit(x,y)
    print(max_features, "option")
    roc = roc_auc_score(y, model.oob_prediction_)
    print("C-stat: ", roc)
    results.append(roc)
    print ("")
    
pd.Series(results, max_features_options).plot(kind="barh", xlim=(0.85,0.88));


results = []
min_samples_leaf_options = [1,2,3,4,5,6,7,8,9,10]

for min_samples in min_samples_leaf_options:
    model = RandomForestRegressor(n_estimators=1000, 
                                  oob_score=True, 
                                  n_jobs=-1, 
                                  random_state=42, 
                                  max_features="auto",
                                  min_samples_leaf=min_samples)
    model.fit(x,y)
    print(min_samples, "min samples")
    roc = roc_auc_score(y, model.oob_prediction_)
    print("C-stat: ", roc)
    results.append(roc)
    print ("")
    
pd.Series(results, min_samples_leaf_options).plot();


#Final Model

model = RandomForestRegressor(n_estimators=1000, 
                                  oob_score=True, 
                                  n_jobs=-1, 
                                  random_state=42, 
                                  max_features="log2",
                                  min_samples_leaf=5)
model.fit(x,y)
roc = roc_auc_score(y, model.oob_prediction_)
print("C:stat", roc)


#Preparing test data set to make sure it aligns with training dataset x
test["Age"].fillna(test.Age.mean(), inplace=True)
test.drop(["Name", "Ticket"], axis=1, inplace=True)
test["Cabin"] = test.Cabin.apply(clean_cabin)
test["Fare"].fillna(test.Fare.mean(), inplace=True)
test["Cabin_T"]=0
test["Embarked_Missing"]=0
test.head()

for variable in categorical_variables:
    #Fill missing data with the word 'Missing'
    test[variable].fillna("Missing", inplace=True)
    #Create array of dummies
    dummies = pd.get_dummies(test[variable], prefix=variable)
    #Update test to include dummies and drop the main variable
    test = pd.concat([test, dummies], axis=1)
    test.drop([variable], axis=1, inplace=True)
    

survived = model.predict(test.as_matrix(columns=None))
b = survived >0.47
c = b.astype(int)
pdtest = pd.DataFrame({'PassengerId': test.PassengerId.astype(int),
                            'Survived': c})
pdtest.to_csv('gptest2.csv', index=False)
