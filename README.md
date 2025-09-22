# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. **Data Loading & Preprocessing**  
   - Read the dataset using `pandas.read_csv()`.  
   - Inspect the data with `.head()`, `.info()`, and `.isnull().sum()` to understand structure and check for missing values.  
   - Encode the categorical column `"salary"` to numeric using `LabelEncoder`.

2. **Feature and Target Selection**  
   - Define the feature matrix **X** with columns:  
     `["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]`.  
   - Define the target vector **y** as the `"left"` column.

3. **Model Training & Prediction**  
   - Split the dataset into training and testing sets using `train_test_split`.  
   - Create a `DecisionTreeClassifier` with `criterion="entropy"` and fit it on the training data.  
   - Predict the target labels for the test set using the trained model.

4. **Model Evaluation & Single Prediction**  
   - Compute and display metrics such as `accuracy_score`, `confusion_matrix`, and `classification_report`.  
   - Make a sample prediction for a new employee record using `dt.predict([[0.5,0.8,9,260,6,8,1,2]])`.


## Program:
```python 
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: B Surya Prakash
RegisterNumber:  212224230281
*/

import pandas as pd
df=pd.read_csv("Employee.csv")

df.head()

df.info()

df.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["salary"]=le.fit_transform(df["salary"])

df['left'].value_counts()

x=df[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=df["left"]
y.head()

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
print("Name: B Surya Prakash")
print("Reg No: 212224230281")

y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
print("Name: B Surya Prakash")
print("Reg No: 212224230281")
accuracy

print("Name: B Surya Prakash")
print("Reg No: 212224230281")
confusion_matrix=metrics.confusion_matrix(y_test,y_pred)
confusion_matrix

Classification_report=metrics.classification_report(y_test,y_pred)
print("Name: B Surya Prakash")
print("Reg No: 212224230281")
print(Classification_report)

print("Name: B Surya Prakash")
print("Reg No: 212224230281")
dt.predict([[0.5,0.8,9,260,6,8,1,2]])
```

## Output:

<img width="982" height="158" alt="image" src="https://github.com/user-attachments/assets/baae5cdc-1159-42b5-b8ed-0cbd897752e4" />

<img width="983" height="292" alt="image" src="https://github.com/user-attachments/assets/ee45aafe-6dc8-4cb0-a733-d1d6daccaccb" />

<img width="989" height="198" alt="image" src="https://github.com/user-attachments/assets/cda8d3b4-00eb-4a3c-a374-66c12fcfd1c2" />

<img width="985" height="57" alt="image" src="https://github.com/user-attachments/assets/a6a42ea5-f0ad-4ec1-b9fe-4ca53d122387" />

<img width="975" height="164" alt="image" src="https://github.com/user-attachments/assets/a73475c9-b4dd-444c-870f-f0153065cfe2" />

<img width="987" height="112" alt="image" src="https://github.com/user-attachments/assets/30d4bd93-0ff5-4ff0-9fc8-9fe1cf53fe04" />

<img width="996" height="77" alt="image" src="https://github.com/user-attachments/assets/967e4d97-e70a-420f-990e-9cd90e22498b" />

<img width="990" height="70" alt="image" src="https://github.com/user-attachments/assets/99827845-3839-4165-ac8b-e00aab11c956" />

<img width="997" height="90" alt="image" src="https://github.com/user-attachments/assets/7baaef19-dd55-4bd6-a480-073cd6850d00" />

<img width="990" height="180" alt="image" src="https://github.com/user-attachments/assets/0a854843-4338-4987-b2d6-35d7c1fa8ee5" />

<img width="984" height="139" alt="image" src="https://github.com/user-attachments/assets/784fc8b3-08e6-4f12-a4bc-f0aaac6da886" />



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
