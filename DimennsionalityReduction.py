#!/usr/bin/env python
# coding: utf-8

# In[4]:


# This file contains dimennsionlity reduction (feature_selection) classes


# In[4]:


from sklearn.base import clone
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()

from sklearn.ensemble import RandomForestClassifier


# In[5]:


from sklearn.base import clone
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()

from sklearn.ensemble import RandomForestClassifier

class SBS():
    def __init__(self, estimator, k_features, scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        # split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        # standardise the features 
        X_train, X_test = stdsc.fit_transform(X_train), stdsc.fit_transform(X_test)

        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train, 
                                 X_test, y_test, self.indices_)
        self.scores_ = [score]

        while dim > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indices_, r=dim-1):
                score = self._calc_score(X_train, y_train, 
                                         X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score
    
    def plot_score(self):
        k_feat = [len(k) for k in self.subsets_]
        plt.plot(k_feat, self.scores_, marker='o')
        plt.ylim([0.5, 1.1])
        plt.ylabel('Accuracy')
        plt.xlabel('Number of features')
        plt.grid()
        plt.show() 


# In[ ]:





# In[8]:


'''
def train_test_spliting_and_standardise(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    X_train_std =stdsc.fit_transform(X_train)
    X_test_std = stdsc.fit_transform(X_test)
    return X_train_std, y_train, X_test_std, y_test
    

def SBS_unit_test():
    df_wine = pd.read_csv('wine.data', header=None)
    df_wine.columns=['Class label', 'Alkohol', 'Apfelsäure', 'Asche', 'Aschealkalintät', 'Magnesium', 
                     'Phenole insgesamt', 'Flavanoide', 'Nicht flavanoide Phenole', 'Tannin', 'Farbintensität',
                    'Farbe', 'des verdünnten Weins', 'Prolin']
    # seperate features and labels
    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    # split train and test as well as standardise the features
    X_train_std, y_train, X_test_std, y_test = train_test_spliting_and_standardise(X, y)

    # test the SBS implementation
    knn = KNeighborsClassifier(n_neighbors=2)
    sbs = SBS(knn, k_features=1)
    sbs.fit(X, y)
    sbs.plot_score()
    k5=list(sbs.subsets_[8])
    print(df_wine.columns[1:][k5])
    
SBS_unit_test()
'''


# In[6]:


from sklearn.ensemble import RandomForestClassifier

class RFC():
    '''
    RandomForestClassifier
    '''
    def __init__(self, feat_labels, n_estimators=10000, random_state=0, n_jobs=-1):
        self.feat_labels = feat_labels
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs

        
    def fit(self, X, y):
        forest = RandomForestClassifier(n_estimators=self.n_estimators, 
                                        random_state=self.random_state, 
                                        n_jobs=self.n_jobs)
        forest.fit(X, y)
        importances = forest.feature_importances_

        self.indices = np.argsort(importances)[::-1]

        for f in range(X.shape[1]):
                print("%2d) %-*s %f" % (f + 1, 30,
                                    feat_labels[self.indices[f]],
                                    importances[self.indices[f]]))

    def plot(self, X):
        plt.title('Feature Importances')
        plt.bar(range(X.shape[1]),
                importances[self.indices],
                color='lightblue',
                align='center')

        plt.xticks(range(X.shape[1]), self.feat_labels[self.indices], rotation=90)
        plt.xlim([-1, X.shape[1]])
        plt.show()


# In[ ]:




