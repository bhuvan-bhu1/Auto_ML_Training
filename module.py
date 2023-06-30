from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score,accuracy_score
import numpy as np
import pandas as pd
import joblib
import pickle


from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.svm import SVC,SVR
from sklearn.neighbors import KNeighborsClassifier


#Regression
def LinearRegression_train(dataset,independent,dependent,size):
    model = LinearRegression()
    df = pd.read_csv(dataset)
    X = df.drop(list(set(df.columns) - set(independent)),axis = 1).values
    y = df[dependent]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=size/100,random_state=10)
    model.fit(X_train,y_train)
    joblib.dump(model,'LinearRegression-joblib')
    pickle.dump(model, open('LinearRegression.pkl', 'wb'))
    y_pred = model.predict(X_test)
    accuracy = r2_score(y_test,y_pred)
    return round(accuracy,5)

def RandomForestRegressor_train(dataset,independent,dependent,size):
    model = RandomForestRegressor(n_estimators=10)
    df = pd.read_csv(dataset)
    X = df.drop(list(set(df.columns) - set(independent)),axis = 1).values
    y = df[dependent]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=size/100,random_state=10)
    model.fit(X_train,y_train)
    joblib.dump(model,'RandomForestRegressor-joblib')
    pickle.dump(model, open('RandomForestRegressor.pkl', 'wb'))
    y_pred = model.predict(X_test)
    accuracy = r2_score(y_test,y_pred)
    return round(accuracy,5)

def DecisionTreeRegressor_train(dataset,independent,dependent,size):
    model = DecisionTreeRegressor()
    df = pd.read_csv(dataset)
    X = df.drop(list(set(df.columns) - set(independent)),axis = 1).values
    y = df[dependent]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=size/100,random_state=10)
    model.fit(X_train,y_train)
    joblib.dump(model,'DecisionTreeRegressor-joblib')
    pickle.dump(model, open('DecisionTreeRegressor.pkl', 'wb'))
    y_pred = model.predict(X_test)
    accuracy = r2_score(y_test,y_pred)
    return round(accuracy,5)

def SupportVectorRegressor_train(dataset,independent,dependent,size):
    model = SVR()
    df = pd.read_csv(dataset)
    X = df.drop(list(set(df.columns) - set(independent)),axis = 1).values
    y = df[dependent]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=size/100,random_state=10)
    model.fit(X_train,y_train)
    joblib.dump(model,'SupportVectorRegressor-joblib')
    pickle.dump(model, open('SupportVectorRegressor.pkl', 'wb'))
    y_pred = model.predict(X_test)
    accuracy = r2_score(y_test,y_pred)
    return round(accuracy,5)

#Classification
def LogisticRegression_train(dataset,independent,dependent,size):
    model = LogisticRegression()
    df = pd.read_csv(dataset)
    X = df.drop(list(set(df.columns) - set(independent)),axis = 1).values
    y = df[dependent]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=size/100,random_state=10)
    model.fit(X_train,y_train)
    joblib.dump(model,'LogisticRegression-joblib')
    pickle.dump(model, open('LogisticRegression.pkl', 'wb'))
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    return round(accuracy,5)

def RandomForestClassifier_train(dataset,independent,dependent,size):
    model = RandomForestClassifier()
    df = pd.read_csv(dataset)
    X = df.drop(list(set(df.columns) - set(independent)),axis = 1).values
    y = df[dependent]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=size/100,random_state=10)
    model.fit(X_train,y_train)
    joblib.dump(model,'RandomForestClassifier-joblib')
    pickle.dump(model, open('RandomForestClassifier.pkl', 'wb'))
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    return round(accuracy,5)

def DecisionTreeClassifier_train(dataset,independent,dependent,size):
    model = DecisionTreeClassifier()
    df = pd.read_csv(dataset)
    X = df.drop(list(set(df.columns) - set(independent)),axis = 1).values
    y = df[dependent]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=size/100,random_state=10)
    model.fit(X_train,y_train)
    joblib.dump(model,'DecisionTreeClassifier-joblib')
    pickle.dump(model, open('DecisionTreeClassifier.pkl', 'wb'))
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    return round(accuracy,5)

def SupportVectorClassifier_train(dataset,independent,dependent,size):
    model = SVC()
    df = pd.read_csv(dataset)
    X = df.drop(list(set(df.columns) - set(independent)),axis = 1).values
    y = df[dependent]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=size/100,random_state=10)
    model.fit(X_train,y_train)
    joblib.dump(model,'SupportVectorClassifier-joblib')
    pickle.dump(model, open('SupportVectorClassifier.pkl', 'wb'))
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    return round(accuracy,5)

def KNeighborsClassifier_train(dataset,independent,dependent,size):
    model = KNeighborsClassifier(n_neighbors=10)
    df = pd.read_csv(dataset)
    X = df.drop(list(set(df.columns) - set(independent)),axis = 1).values
    y = df[dependent]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=size/100,random_state=10)
    model.fit(X_train,y_train)
    joblib.dump(model,'KNeighborsClassifier-joblib')
    pickle.dump(model, open('KNeighborsClassifier.pkl', 'wb'))
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    return round(accuracy,5)