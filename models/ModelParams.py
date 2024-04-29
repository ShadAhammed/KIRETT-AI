from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import numpy as np


SVCParams= {
            'kernel': ['rbf'],
            'gamma': [0.1, 0.01,0.001],
            'C': [1.0, 0.1, 10],
        }
SVCModel = SVC(probability=True)

XGBParams = {
        'min_child_weight': [3, 5, 8],
        'gamma': [0.8, 1, 1.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [5,7]
        }

XGBModel = XGBClassifier()

RFParams = {
        'n_estimators' : [50,100,120],
        'max_samples': [0.6, 0.8],
        'max_depth': [3, 4],
        'criterion': ['gini', 'entropy'],
        'min_samples_split': [3, 5, 8]
        }

RFModel = RandomForestClassifier()

LRParams = {"C":np.logspace(-3,3,7), "penalty":["l1","l2"], 'solver'  : ['newton-cg', 'lbfgs', 'liblinear']}
LRModel= LogisticRegression()

NBParams = {'var_smoothing'  : [1e-09, 1e-08]}
NBModel= GaussianNB()

n_neighbors = range(1, 10, 2)
weights = ['uniform', 'distance']
metric = ['euclidean', 'manhattan', 'minkowski']


KNNParams = dict(n_neighbors=n_neighbors,weights=weights,metric=metric)
KNNModel= KNeighborsClassifier()