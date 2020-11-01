import pandas as pd
import numpy as np

import time
import math
from sklearn.ensemble import RandomForestClassifier
from MIRCO import MIRCO
from MIRCO import MIRCOfit
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

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

#divide into train and test sets for holdout
trainX, testX, trainy, testy = \
    train_test_split(X, y, test_size = 0.3, random_state = randomstate)

#MIRCO
MIRCO.greedySCP = greedySCPadj #replace function
RF = RandomForestClassifier(random_state = randomstate, criterion = crit, max_depth = tree_depth
                            ) #create decision tree classifier object
RF_fit = RF.fit(trainX, trainy) #train decision tree classifier
MRC = MIRCO(RF_fit)
MRC_fit = MRC.fit(trainX, trainy)
MRC_pred = MRC_fit.predict(testX) #predict response for test dataset
end = time.time()


def exportRulesadj(self):
        for rindx, rule in enumerate(self.rules.values()):
            print('RULE %d:' % rindx)
            
            #construct matrix containing all the rules
            ruleMat = np.array(rule[:-1])
           
            #find which unique features are used for this rule
            uniqueValues = np.unique(ruleMat[:, 0])
            #the amount of unique features
            uniqueAmount = len(uniqueValues)
            #construct a matrix with the number of the feature in the centre, - infinity left and infinity right
            simpMatrix = np.zeros(3 * uniqueAmount).reshape((uniqueAmount, 3))
            for ruleNumber in range(uniqueAmount):
                simpMatrix[ruleNumber, 0] = -np.Inf
                simpMatrix[ruleNumber, 1] = uniqueValues[ruleNumber]
                simpMatrix[ruleNumber, 2] = np.Inf
            
            #fill the previously constructed matrix using the given rules
            for i in range(uniqueAmount):
                for j in range(len(ruleMat)):
                    if (simpMatrix[i, 1] == int(ruleMat[j, 0])):
                        if ((ruleMat[j, 1] == 'l') and float(ruleMat[j, 2]) < simpMatrix[i, 2]):
                            simpMatrix[i, 2] = math.floor(float(ruleMat[j, 2]))
                        if ((ruleMat[j, 1] == 'r') and float(ruleMat[j, 2]) > simpMatrix[i, 0]):
                            simpMatrix[i, 0] = math.floor(float(ruleMat[j, 2]))
            
            #print the results in a compact way
            for i in range(uniqueAmount):
                if ((simpMatrix[i, 0] != -np.Inf) and simpMatrix[i, 2] == np.Inf):
                    print('==> x[',int(simpMatrix[i, 1]),'] >', simpMatrix[i, 0])
                elif ((simpMatrix[i, 0] == -np.Inf) and simpMatrix[i, 2] != np.Inf):
                    print('==> x[',int(simpMatrix[i, 1]),'] <=', simpMatrix[i, 2])
                elif ((simpMatrix[i, 0] != -np.Inf) and simpMatrix[i, 2] != np.Inf):
                    print('==>', simpMatrix[i, 0], '< x[', int(simpMatrix[i, 1]),'] <=', simpMatrix[i, 2])
                
            #give the class numbers
            strarray = '['
            for cn in rule[-1][0:-1]:
                strarray += ('{0:.2f}'.format(cn) + ', ')
            strarray += ('{0:.2f}'.format(rule[-1][-1]) + ']')
                
            print('==> Class numbers: %s' % strarray) 
            

print('\n\nRules obtained by MIRCO')
MIRCOfit.exportRules = exportRulesadj #replace function
rules_MRC = MRC_fit.exportRules()