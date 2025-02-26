# Ex 02 - Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas. 




## Program With Output:
```
*/
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: JAYAKUMAR B 
RegisterNumber:  212223040073
*/
```

```
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

dataset = pd.read_csv("/student_scores.csv")
print(dataset.head())
print(dataset.tail())
```

![image](https://github.com/user-attachments/assets/f123adb8-4d47-43a7-942f-24bce03dcb9d)


```
dataset.info()
```

![image](https://github.com/user-attachments/assets/e8dfb0d8-0989-4ae5-98bd-c87b957c4f55)


```
dataset.describe()
```

![image](https://github.com/user-attachments/assets/aef74202-8107-4695-8280-a99369d7ab9c)


```
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,-1].values
print(y)
```

![image](https://github.com/user-attachments/assets/9217e5aa-9ef3-4326-aab9-5d9bc53a8583)


```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
x_train.shape
```

![image](https://github.com/user-attachments/assets/dd68693d-e1ec-436c-b1ea-d5f2cc8b3952)


```
x_test.shape
```

![image](https://github.com/user-attachments/assets/e58784d2-17b6-4904-9476-13b051eb4e86)


```
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
```

![image](https://github.com/user-attachments/assets/ffcd3291-5a3c-496b-a4e1-8046ecb2fd76)


```
y_pred=reg.predict(x_test)
print(y_pred)
print(y_test)
```

![image](https://github.com/user-attachments/assets/73b8706a-4dda-4865-bc9d-5c5bd026086f)


```
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```

![image](https://github.com/user-attachments/assets/f152c704-7bfe-4eb6-b520-e17c73e90e98)


```
plt.scatter(x_train,y_train,color="green")
plt.plot(x_train,reg.predict(x_train),color="red")
plt.title('Training set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(x_test,y_test,color="blue")
plt.plot(x_test,reg.predict(x_test),color="silver")
plt.title('Test_set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```

![image](https://github.com/user-attachments/assets/c83531ee-54cb-4f48-b8dd-7e4b52592a51)

![image](https://github.com/user-attachments/assets/558ab2a5-dd7b-4820-af0b-1b65daeb7c32)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
