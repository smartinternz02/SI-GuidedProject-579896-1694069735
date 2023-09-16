# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 12:51:03 2023

@author: Acer
"""

#pandas
import pandas as pd 

#matplotlib
import matplotlib.pyplot as plt 

#seaborn
import seaborn as sns

#numpy
import numpy as np

#sklearn
from sklearn.metrics import confusion_matrix,classification_report

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

import pickle


##Reading Data Set

crop_df = pd.read_csv("C:/Users\Acer/Desktop/SMARTBRIDGE/PROJECTS/Crop_recommendation.csv")
crop_df.head()

##Check Null values are there or not

crop_df.isnull().sum()

crop_df['no_label'] = pd.Categorical(crop_df.label)
#no_label=['rice','maize','chickpea','kidneybeans','pigeonpeas','mothbeans','mungbean','blackgram','lentil','pomegranate','banana','mango','grapes'.'watermelon','muskmelon','apple','orange','papaya','coconut','cotton','jute','coffee']



##EDA
##Univariate Analysis

#Nitrogen
plt.figure(figsize=(8,7))
sns.histplot(x='N',data=crop_df,color='b');
plt.title("Nitrogen for crops",{'fontsize':20}); 

#Potassium
plt.figure(figsize=(8,7))
sns.histplot(x='K',data=crop_df,color='b');
plt.title("Potassium for crops",{'fontsize':20});

#Phosphorus
plt.figure(figsize=(8,7))
sns.histplot(x='P',data=crop_df,color='b');
plt.title("Phosphorus for crops",{'fontsize':20});

crop_df.columns

#Temperature
plt.figure(figsize=(10,6))
sns.boxplot(x=crop_df.temperature);

#Humidity
plt.figure(figsize=(10,6))
sns.boxplot(x=crop_df.humidity);

#PH
plt.figure(figsize=(8,7))
sns.histplot(x='ph',data=crop_df,color='b');
plt.title("PH for crops",{'fontsize':20});

#Rainfall
plt.figure(figsize=(8,7))
sns.histplot(x='rainfall',data=crop_df,color='b');
plt.title("Rainfall feature",{'fontsize':20});

#Distplot
sns.distplot(crop_df['ph'])


##Bivariate analysis
sns.scatterplot(crop_df["temperature"],crop_df["no_label"])
sns.FacetGrid(crop_df,hue="no_label",size=5).map(plt.scatter,"N","K").add_legend();
sns.FacetGrid(crop_df,hue="no_label",size=5).map(plt.scatter,"N","P").add_legend()
sns.FacetGrid(crop_df,hue="no_label",size=5).map(plt.scatter,"P","K").add_legend()
sns.FacetGrid(crop_df,hue="no_label",size=5).map(plt.scatter,"N","temperature").add_legend()
sns.FacetGrid(crop_df,hue="no_label",size=5).map(plt.scatter,"N","humidity").add_legend()
sns.FacetGrid(crop_df,hue="no_label",size=5).map(plt.scatter,"N","ph").add_legend()
plt.show()
#Multivariate analysis

sns.pairplot(crop_df,hue="no_label",size=3)

#get correlations of each features in dataset
corrmat =crop_df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(crop_df[top_corr_features].corr(),annot=True,cmap="RdYlGn")

#Split data

X = crop_df.drop(['label','no_label'],axis=1)
#from sklearn.preprocessing import LabelEncoder
#le=LabelEncoder()
#y=le.fit_transform(crop_df['label'])
y= pd.Categorical(crop_df.label)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

X_train.head()

##Feature Scaling
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)
y=crop_df.no_label
##Building models

# Support Vector Machines Classifier
svm = SVC()
svm.fit(X_train, y_train)
print("The accuracy of SVM is",
      svm.score(X_train,y_train), svm.score(X_test,y_test))
svm = [svm.score(X_train,y_train), svm.score(X_test,y_test)]



#Random Forest Classifier Model
rfclassifier = RandomForestClassifier()
rfclassifier.fit(X_train, y_train)
print("The accuracy of random forest Classifier is",
      rfclassifier.score(X_train,y_train), rfclassifier.score(X_test,y_test))
rf = [rfclassifier.score(X_train,y_train), rfclassifier.score(X_test,y_test)]



#Decision Tree Classifier Model
dtclassifier = DecisionTreeClassifier(max_depth=7)
dtclassifier.fit(X_train,y_train)
print("The accuracy of Decision Tree Classifier is",
      dtclassifier.score(X_train,y_train),dtclassifier.score(X_test,y_test))
dt = [dtclassifier.score(X_train,y_train),dtclassifier.score(X_test,y_test)]

# K-Nearest-Neighbors Classifier
knnclassifier = KNeighborsClassifier(n_neighbors=9)
knnclassifier.fit(X_train, y_train)
print("The accuracy of K Nearest Neighbors Classifier is",
      knnclassifier.score(X_train,y_train), knnclassifier.score(X_test,y_test))
knn = [knnclassifier.score(X_train,y_train), knnclassifier.score(X_test,y_test)]



#Results table for comparison of accuracies
results1 = pd.DataFrame(data=[svm,rf,dt,knn],
                        columns = ['Training Accuracy ', 'Testing Accuracy '],
                        index = ['Support Vector Machine',
                                  'Random Forest','Decision tree','knn'])

knnclassifier =KNeighborsClassifier()
knnclassifier.fit(X_train,y_train)

y_pred =knnclassifier.predict(X_test)

print(classification_report(y_test,y_pred))

##confusion matrix
class_names = np.arange(0,21)
fig,ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks,class_names)
plt.yticks(tick_marks,class_names)
cnf_matrix = confusion_matrix(y_test,y_pred)
sns.heatmap(pd.DataFrame(cnf_matrix), annot = True,fmt = 'd')
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.title('Confusion Matrix for knn is', {'fontsize':20})
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()



# open a file, where you want to store the data
pickle.dump(knnclassifier, open('model.pkl','wb'))
# dump information to that file

print(knnclassifier.predict([[90,42,43,20.87,82,6.5,202.93]]))