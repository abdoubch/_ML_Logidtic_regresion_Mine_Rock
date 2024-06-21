# Dependencies
"""
# print(dataset.describe()) gives info like mean , std , count ... about data 

# print(dataset[N].value_counts()) group rows according to N-th column value and count them 

# print(dataset.groupby(Nth).mean()) group according to the Nth column values and calculating mean for mean to each column that got values of the same group 
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split # used for splliting data in two parts : training data and testing data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 

# data collection
# Loading dataset :
dataset = pd.read_csv('./data/dataset.csv',header=None)   

# Separing Data from labels
X = dataset.drop(columns=60,axis=1)
Y = dataset[60]

# split data into training data and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=1)  


# Training model --> Logostic regression
model = LogisticRegression()

model.fit(X_train,Y_train)


'''
 Evaluating model:
 Accuracy on training data :
 X_train_prediction = model.predict(X_train)
 training_data_accuracy = accuracy_score(X_train_prediction,Y_train)
 print(f"Accuracy on training data : {training_data_accuracy}")
 Accuracy on testing data :
 X_test_prediction = model.predict(X_test)
 testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)
 print(f"Accuracy on testing data : {testing_data_accuracy}")
'''

def predicting_system(data):
    data_as_array = np.asarray(data)
    data_reshaped = data_as_array.reshape(1,-1)
    if(model.predict(data_reshaped)[0] == 'R'):
        return'The obkect is a Rock'
    else:
        return "The object is a Mine"

# Testing on an exemple 
input_data = (
    0.0414,
    0.0436,
    0.0447,
    0.0844,
    0.0419,
    0.1215,
    0.2002,
    0.1516,
    0.0818,
    0.1975,
    0.2309,
    0.3025,
    0.3938,
    0.5050,
    0.5872,
    0.6610,
    0.7417,
    0.8006,
    0.8456,
    0.7939,
    0.8804,
    0.8384,
    0.7852,
    0.8479,
    0.7434,
    0.6433,
    0.5514,
    0.3519,
    0.3168,
    0.3346,
    0.2056,
    0.1032,
    0.3168,
    0.4040,
    0.4282,
    0.4538,
    0.3704,
    0.3741,
    0.3839,
    0.3494,
    0.4380,
    0.4265,
    0.2854,
    0.2808,
    0.2395,
    0.0369,
    0.0805,
    0.0541,
    0.0177,
    0.0065,
    0.0222,
    0.0045,
    0.0136,
    0.0113,
    0.0053,
    0.0165,
    0.0141,
    0.0077,
    0.0246,
    0.0198,
)

print(predicting_system(input_data))
