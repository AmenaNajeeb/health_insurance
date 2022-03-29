#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import pickle

#pip install imbalanced-learn

from imblearn.over_sampling import SMOTE

df=pd.read_csv("C:\\Users\\Amena\\Desktop\\corporate_training_excelr\\a_model_deploy\\Insurance_Dataset.csv")

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

#from sklearn.tree import DecisionTreeClassifier

#dt0 = DecisionTreeClassifier(random_state=0)

#dt0.fit(X_train, y_train)

#y_pred = dt0.predict(X_test)

#print("Training Accuracy :", dt0.score(X_train, y_train))
#print("Testing Accuracy :", dt0.score(X_test, y_test))

#print(cross_val_score(dt0, X, y, cv=10, scoring = "accuracy").mean())

from sklearn.metrics import classification_report
#print(classification_report(y_test, y_pred))

#above model has overfitting problem, so we will tune the parameters.

from sklearn.tree import DecisionTreeClassifier

#dt1 = DecisionTreeClassifier(max_depth=10,random_state=0)

#dt1.fit(X_train, y_train)

#y_pred = dt1.predict(X_test)

#print("Training Accuracy :", dt1.score(X_train, y_train))
#print("Testing Accuracy :", dt1.score(X_test, y_test))

#print(cross_val_score(dt0, X, y, cv=10, scoring = "accuracy").mean())

#print(classification_report(y_test, y_pred))

# When tuning, the max depth parameter is used here, it sets the maximum depth of the tree as 10. As we are pruning (cutting down) the tree, the accuracy decreases. But the overfitting problem is solved i.e., training accuracy is close to testing accuracy.

#dt3 = DecisionTreeClassifier(criterion= 'gini', min_samples_leaf=5, min_samples_split= 10, max_depth=25)

#dt3.fit(X_train, y_train)

#y_pred = dt3.predict(X_test)

#print("Training Accuracy :", dt3.score(X_train, y_train))
#print("Testing Accuracy :", dt3.score(X_test, y_test))

#print(classification_report(y_test, y_pred))

dt4 = DecisionTreeClassifier(max_leaf_nodes=1000)

dt4.fit(X_train, y_train)

y_pred = dt4.predict(X_test)

print("Training Accuracy :", dt4.score(X_train, y_train))
print("Testing Accuracy :", dt4.score(X_test, y_test))

print(classification_report(y_test, y_pred))


# Saving model to disk
#pickle.dump(dt4, open('model.pkl','wb'))

# Loading model to compare the results
#model = pickle.load(open('model.pkl','rb'))
#print(model.predict([[2, 9, 6]]))

# from sklearn.ensemble import RandomForestClassifier


# In[27]:


# model = RandomForestClassifier()
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)

# print("Training Accuracy :", model.score(X_train,y_train))
# print("Testing Accuracy :", model.score(X_test, y_test))

# # cm = confusion_matrix(y_test, y_pred)
# # print(cm)


# In[28]:


# print(classification_report(y_test, y_pred))


# Testing Accuracy with Random Forest algorithm = 75% without hyperparameter tuning.
# 
# Problem: overfitting+long time to execute when compared to Decision Tree.

# In[29]:


# from sklearn.linear_model import LogisticRegression


# In[30]:


# model_log = LogisticRegression()
# model_log.fit(X_train,y_train)
# y_pred = model_log.predict(X_test)


# In[31]:


# print(classification_report(y_test, y_pred))


# Accuracy is 50%.

# In[32]:


# from sklearn.linear_model import SGDClassifier


# In[33]:


# s1 = SGDClassifier()


# In[34]:


# s1.fit(X_train, y_train)


# In[35]:


# y_pred_s = s1.predict(X_test)


# In[36]:


# print("Training Accuracy :", s1.score(X_train, y_train))
# print("Testing Accuracy :", s1.score(X_test, y_test))


# Accuracy here is 50%.

# In[37]:


# print(classification_report(y_test, y_pred_s))


# In[ ]:





# ### Conclusion
# 
# The model dt4 can be considered for deployment.

# In[ ]:




