import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from statsmodels.stats.contingency_tables import mcnemar

dataset = pd.read_excel("C:\\Users\\marj9\\Downloads\\ass5.xls")
data = dataset.values

#divide in explanatory and dependent variables
X = data[:, 1:24]
y = data[:, 0]

#fix random seed
randomstate = 99

#divide into train and test set
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size = 0.3, random_state = randomstate)

#polynomial kernel
param_grid_poly = {'C': [0.1, 1,  10, 100, 1000], 'gamma': [0.0001, 0.01, 1], 'degree': [0, 1, 2, 3], 'kernel': ['poly']}
gridpoly = GridSearchCV(SVC(random_state = randomstate), param_grid_poly, verbose = 2)
gridpoly.fit(X_train, y_train)
gridpoly_best_gamma = gridpoly.best_estimator_.gamma
gridpoly_best_c = gridpoly.best_estimator_.C
gridpoly_best_degree = gridpoly.best_estimator_.degree

clf = SVC(random_state = randomstate, kernel = 'poly', cache_size = 1500, gamma = gridpoly_best_gamma, C = gridpoly_best_c, degree = gridpoly_best_degree)
clf.fit(X_train, y_train)
y_predp = clf.predict(X_test)
acc_poly = accuracy_score(y_predp, y_test)
poly_confusion = confusion_matrix(y_test, y_predp)
poly_classification = classification_report(y_test, y_predp) 

#radial kernel 
param_grid_rad = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [0.0001, 0.01, 1], 'kernel': ['rbf']}
gridrad = GridSearchCV(SVC(random_state = randomstate), param_grid_rad, verbose = 2)
gridrad.fit(X_train, y_train)
gridrad_best_gamma = gridrad.best_estimator_.gamma
gridrad_best_c = gridrad.best_estimator_.C

clf = SVC(random_state = randomstate, kernel = 'rbf', cache_size = 1500, gamma = gridrad_best_gamma, C = gridrad_best_c)
clf.fit(X_train, y_train)
y_predr = clf.predict(X_test)
acc_rad = accuracy_score(y_predr, y_test)  
rad_confusion = confusion_matrix(y_test, y_predr)
rad_classification = classification_report(y_test, y_predr) 

#sigmoid kernel 
param_grid_sig = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [0.0001, 0.01, 1], 'kernel': ['sigmoid']}
gridsig = GridSearchCV(SVC(random_state = randomstate), param_grid_sig, verbose = 2)
gridsig.fit(X_train, y_train)
gridsig_best_gamma = gridsig.best_estimator_.gamma
gridsig_best_c = gridsig.best_estimator_.C

clf = SVC(random_state = randomstate, kernel = 'sigmoid', cache_size = 1500, gamma = gridsig_best_gamma, C = gridsig_best_c)
clf.fit(X_train, y_train)
y_preds = clf.predict(X_test)
acc_sig = accuracy_score(y_preds, y_test)
sig_confusion = confusion_matrix(y_test, y_preds)
sig_classification = classification_report(y_test, y_preds) 

#contingency table
def contingency(y_pred1, y_pred2):
    correct1 = np.zeros(len(y_pred1))
    for i in range(len(correct1)):
        if (y_pred1[i] == y_test[i]):
            correct1[i] = 1
    correct2 = np.zeros(len(y_pred2))
    for i in range(len(correct2)):
        if (y_pred2[i] == y_test[i]):
            correct2[i] = 1
    table = np.zeros((2, 2))
    for i in range(len(y_pred1)):
        if (correct1[i] == 1 and correct2[i] == 1):
            table[0, 0] = table[0, 0] + 1
        elif (correct1[i] == 1 and correct2[i] == 0):
            table[0, 1] = table[0, 1] + 1
        elif (correct1[i] == 0 and correct2[i] == 1):
            table[1, 0] = table[1, 0] + 1
        else:
            table[1, 1] = table[1, 1] + 1
    return table
 
table = contingency(y_predp, y_predr)
result = mcnemar(table, exact = True)
print('statistic=%.3f, p-value = %.3f' % (result.statistic, result.pvalue))
table = contingency(y_predp, y_preds)
result = mcnemar(table, exact = True)
print('statistic=%.3f, p-value = %.3f' % (result.statistic, result.pvalue))
table = contingency(y_predr, y_preds)
result = mcnemar(table, exact = True)
print('statistic=%.3f, p-value = %.3f' % (result.statistic, result.pvalue))
