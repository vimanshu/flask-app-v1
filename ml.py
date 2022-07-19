import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle

df = pd.read_csv('diabetes.csv')

X = df.drop('Outcome', axis =1)
y = df['Outcome']

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2)

knn = KNeighborsClassifier()

knn.fit(X_train,y_train)

# add the weights in a pickle file

pickle.dump(knn,open('model_weight.pkl','wb'))


