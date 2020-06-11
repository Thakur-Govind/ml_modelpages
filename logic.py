from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,fbeta_score,r2_score
import numpy as np
import pandas as pd
def X_y_sep(alpha, X, y):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = alpha, random_state = 42)
    return X_train,X_test,y_train,y_test
def svc_ml(alpha, X, y, c, g):
    X_train,X_test,y_train,y_test = X_y_sep(alpha, X, y)
    model = SVC(C=c, gamma = g)
    model.fit(X_train,y_train)
    labels = model.predict(X_test)
    return labels,y_test
def lr_ml(alpha, X, y):
    X_train,X_test,y_train,y_test = X_y_sep(alpha, X, y)
    model = LinearRegression()
    model.fit(X_train,y_train)
    labels = model.predict(X_test)
    return labels, y_test
def dt_ml(alpha, X, y, m_d, m_s_l):
    X_train,X_test,y_train,y_test = X_y_sep(alpha, X, y)
    model = DecisionTreeClassifier(max_depth = m_d, min_samples_leaf = m_s_l)
    model.fit(X_train,y_train)
    labels = model.predict(X_test)
    return labels, y_test
def rf_ml(alpha, X, y, n_comps, max_d, min_s_s):
    X_train,X_test,y_train,y_test = X_y_sep(alpha, X, y)
    model = RandomForestClassifier(n_estimators = n_comps, max_depth = max_d, min_samples_split = min_s_s,random_state= 24)
    model.fit(X_train,y_train)
    labels = model.predict(X_test)
    return labels, y_test
def adb_ml(alpha,X,y,n_est,lr):
    X_train,X_test,y_train,y_test = X_y_sep(alpha, X, y)
    model = AdaBoostClassifier(n_estimators = n_est, learning_rate = lr, random_state = 42)
    model.fit(X_train,y_train)
    labels = model.predict(X_test)
    return labels, y_test
def tmetrics(y,y_test,class_or_reg):
    if class_or_reg == "class":
        return (accuracy_score(y,y_test),f1_score(y,y_test),fbeta_score(y,y_test,beta = 0.001),fbeta_score(y,y_test, beta = 5),-1.00)
    else:
        return(-1.00,-1.00,-1.00,-1.00,r2_score(y,y_test))
def get_data(file):
    df = pd.read_csv(file)
    a = list(df.columns)
    a.remove('Unnamed: 0')
    a.remove('Label')
    X = df[a]
    y = df['Label']
    return (X,y)
