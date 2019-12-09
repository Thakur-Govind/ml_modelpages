from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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
    model = RandomForestClassifier(n_estimators = n_comps, max_depth = max_d, min_samples_split = min_s_s)
    model.fit(X_train,y_train)
    labels = model.predict(X_test)
    return labels, y_test
def acc(y,y_test):
    return accuracy_score(y,y_test)
