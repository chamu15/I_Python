import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from MIRCO import MIRCO
import time
from sklearn import tree
from sklearn.metrics import accuracy_score

dataset = pd.read_csv("C:\\Users\\marj9\\Downloads\\german.data-numeric.csv")

data = dataset.values

#new greedySCP
def greedySCPadj(self, c, A):
        # Mathematical model
        # minimize     c'x
        # subject to   Ax >= 1
        #              x in {0,1}
        # c: n x 1
        # A: m x n
        
        # number of rows and number of columns
        m, n = A.shape
        # set of rows (items)
        M = set(range(m))
        
        #count for each rule how many samples it covers
        coverSum = np.zeros(n)
        for rule in range(n):
            coverSum[rule] = c[rule]/np.sum(A[:, rule])
        
        #sort the rules such that those with the lowest ratio come first
        NcS = np.stack((np.arange(n), coverSum), axis = 0)
        NcS = np.transpose(NcS)
        sorted_NcS = NcS[np.argsort(NcS[:,1])]
        #sorted set of rules
        newN = np.transpose(sorted_NcS[:, 0])

        R = M
        S = set()
        nDiffS = n #for taking into account all rules instead of subset
        while (len(R) > 0):
            minratio = np.Inf
            for j in range(1000):
                jj = int(newN[j])
                # Sum of covered rows by column j
                denom = np.sum(A[list(R), jj])
                if (denom == 0):
                    continue
                ratio = c[jj]/denom
                if (ratio < minratio):
                    minratio = ratio
                    jstar = jj
            column = A[:, jstar]
            Mjstar = set(np.where(column.toarray() == 1)[0])
            R = R.difference(Mjstar)
            S.add(jstar)
            nDiffS = nDiffS - 1 #if all rules taken into account, one is deleted
            newN1 = newN[0:jstar]
            newN2 = newN[jstar+1:len(newN)]
            newN = np.concatenate((newN1, newN2))

        listS = list(S)
        # Sort indices
        sindx = list(np.argsort(c[listS]))
        S = set()
        totrow = np.zeros((m, 1), dtype=np.int32)
        for i in sindx:
            S.add(listS[i])
            column = A[:, listS[i]]
            totrow = totrow + column
            if (np.sum(totrow > 0) >= m):
                break
        return S

#fix random seed
randomstate = 99

#impurity criterion
crit = "gini"

#divide data set
X = data[:, 0:24]
y = data[:, 24]

#tree depth
tree_depth = 7

#k-fold cross-validation
k = 10
trainSize = (k - 1)/k * len(X)
testSize = int(len(X) - trainSize)

accuracy_DT = np.zeros(k)
accuracy_RF = np.zeros(k)
accuracy_MRC = np.zeros(k)

rulesAmount_DT = np.zeros(k)
rulesAmount_RF = np.zeros(k)
rulesAmount_MRC = np.zeros(k)

missedSamples = np.zeros(k)

timeVec = np.zeros(k)

skf = StratifiedKFold(n_splits = k, shuffle = True, random_state = randomstate)
i = 0
for train_index, test_index in skf.split(X, y):
    xtr, xte = X[train_index], X[test_index]
    ytr, yte = y[train_index], y[test_index]
    
    #decision tree 
    DT = tree.DecisionTreeClassifier(random_state = randomstate, criterion = crit, max_depth = tree_depth) #create decision tree classifier object
    DT_fit = DT.fit(xtr, ytr) #train decision tree classifier
    DT_pred = DT_fit.predict(xte) #predict response for test dataset 
    accuracy_DT[i] = accuracy_score(DT_pred, yte)
    
    #random forest
    RF = RandomForestClassifier(random_state = randomstate, criterion = crit, max_depth = tree_depth 
                                    ) #create decision tree classifier object
    RF_fit = RF.fit(xtr, ytr) #train decision tree classifier
    RF_pred = RF_fit.predict(xte) #predict response for test dataset
    accuracy_RF[i] = accuracy_score(RF_pred, yte)
    
    #MIRCO
    start = time.time()
    MIRCO.greedySCP = greedySCPadj #replace function
    RF = RandomForestClassifier(random_state = randomstate, criterion = crit, max_depth = tree_depth 
                                ) #create decision tree classifier object
    RF_fit = RF.fit(xtr, ytr) #train decision tree classifier
    MRC = MIRCO(RF_fit)
    MRC_fit = MRC.fit(xtr, ytr)
    MRC_pred = MRC_fit.predict(xte) #predict response for test dataset
    accuracy_MRC[i] = accuracy_score(MRC_pred, yte)
    missedSamples[i] = MRC_fit.numOfMissed
    end = time.time()
    timeVec[i] = end - start
    
    #amount of rules
    rulesAmount_DT[i] = DT_fit.tree_.n_leaves
    rulesAmount_RF[i] = MRC_fit.initNumOfRules
    rulesAmount_MRC[i] = MRC_fit.numOfRules
    
    i = i + 1
    


print('## AVERAGE ACCURACIES ##')
print('Decision Tree: ', np.sum(accuracy_DT)/k)
print('Random Forest: ', np.sum(accuracy_RF)/k)
print('MIRCO: ', np.sum(accuracy_MRC)/k)

print('\n## AVERAGE NUMBERS OF RULES ##')
print('Decision Tree: ', np.sum(rulesAmount_DT)/k)
print('Random Forest: ', np.sum(rulesAmount_RF)/k)
print('MIRCO: ', np.sum(rulesAmount_MRC)/k)

print('\nAverage number of missed test samples by MIRCO: ', np.sum(missedSamples)/k)

print('\n## AVERAGE COMPUTATION TIME ##')
print('MIRCO: ', np.sum(timeVec)/k)



print('\n\nRules obtained by MIRCO')
MRC_fit.exportRules()

print('\nRules obtained by DT')
rules_DT = tree.export_text(DT_fit)