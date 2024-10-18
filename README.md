# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1. import necessary python library and load the data set
2.do the required data preprocessing and convert the type of features into category
3.declare the theta value as random numbers and define sigmoid,loss,gradient_descent and prediction function
4.calculate accuracy ,prediction and new prediction
```
## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: MARINO SARISHA T
RegisterNumber:  212223240084
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv("Placement_Data.csv")
data=data.drop(["sl_no","salary"],axis=1)
data
![Screenshot 2024-10-18 105123](https://github.com/user-attachments/assets/99a1fbfa-b025-4b89-bacf-58411795ee56)

data["gender"]=(data["gender"]).astype('category')
data["ssc_b"]=(data["ssc_b"]).astype('category')
data["hsc_b"]=(data["hsc_b"]).astype('category')
data["hsc_s"]=(data["hsc_s"]).astype('category')
data["degree_t"]=(data["degree_t"]).astype('category')
data["workex"]=(data["workex"]).astype('category')
data["specialisation"]=(data["specialisation"]).astype('category')
data["status"]=(data["status"]).astype('category')
data.dtypes
![Screenshot 2024-10-18 105202](https://github.com/user-attachments/assets/28a659da-bc3b-49f1-94ce-918059fae11d)

data["gender"]=(data["gender"]).cat.codes
data["ssc_b"]=(data["ssc_b"]).cat.codes
data["hsc_b"]=(data["hsc_b"]).cat.codes
data["hsc_s"]=(data["hsc_s"]).cat.codes
data["degree_t"]=(data["degree_t"]).cat.codes
data["workex"]=(data["workex"]).cat.codes
data["specialisation"]=(data["specialisation"]).cat.codes
data["status"]=(data["status"]).cat.codes
data
![Screenshot 2024-10-18 105241](https://github.com/user-attachments/assets/06170e59-dc0b-4a64-ae82-aa82412816c1)

x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values
theta = np.random.randn(x.shape[1])
Y=y
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def loss(theta,x,Y):
    h= sigmoid(x.dot(theta))
    return -np.sum(Y * np.log(h) + (1-Y) * np.log(1-h))

def gradient_descent(theta,x,Y,alpha,num_iterations):
    m=len(Y)
    for i in range(num_iterations):
        h = sigmoid(x.dot(theta))
        gradient = x.T.dot(h-Y)/m
        theta -= alpha * gradient
    return theta

theta = gradient_descent(theta,x,Y,alpha=0.01,num_iterations=1000)


def predict(theta,x):
    h = sigmoid(x.dot(theta))
    y_pred = np.where(h >= 0.5 ,1 ,0)
    return y_pred

y_pred = predict(theta,x)

accuracy = np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)
![Screenshot 2024-10-18 105443](https://github.com/user-attachments/assets/d798bc49-e5c9-45be-8b41-81aa1580d842)

print("prediction\n",y_pred)
![Screenshot 2024-10-18 105421](https://github.com/user-attachments/assets/75faa7ec-6972-4318-9291-1cc26e294208)

xnew = np.array(([0,87,0,95,0,0,1,0,2,3,2,3]))
yprednew=predict(theta,xnew)
print("New prediction :",yprednew)
![Screenshot 2024-10-18 105401](https://github.com/user-attachments/assets/47b340a3-5b6f-433f-bb85-961d95862406)

```

## Output:
![logistic regression using gradient descent](sam.png)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

