#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import pickle

#pip install imbalanced-learn

from imblearn.over_sampling import SMOTE

df=pd.read_csv("Insurance_Dataset.csv")

df.shape

##df.head(2)

#df.isnull().sum()

df = df.dropna(how="any") #deleting all null values

#df.isnull().sum() #verifying if the dataframe is free of null values.

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
var_mod = df.select_dtypes(include='object').columns
for i in var_mod:
    df[i] = le.fit_transform(df[i])

X = df[['Tot_cost','Tot_charg','Admission_type', 'Payment_Typology','Surg_Description', 'Hospital Id']] #'Cultural_group', , 'Hospital County'
y = df['Result']

# X = df.drop(['Result'], axis=1)
# y = df['Result']

smote = SMOTE() #sampling_strategy='minority'
X_sm, y_sm = smote.fit_resample(X, y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size = 0.15, random_state=3)


# ## Model Building - 1 (All Variables)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

dt4 = DecisionTreeClassifier(max_leaf_nodes=1000)

dt4.fit(X_train, y_train)

y_pred = dt4.predict(X_test)

print("Training Accuracy :", dt4.score(X_train, y_train))
print("Testing Accuracy :", dt4.score(X_test, y_test))

print(classification_report(y_test, y_pred))


# Saving model to disk
filename='model.pkl'
pickle.dump(dt4, open(filename,'wb'))

# Loading model to compare the results
#model = pickle.load(open('model.pkl','rb'))
#print(model.predict([[2, 9, 6]]))
