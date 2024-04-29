import warnings
warnings.filterwarnings('ignore')
from src.models.ModelTuning import *
from models.ModelParams import *
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve


class SupportVector:

    def __init__(self, X_train,y_train):
        self.X_train= X_train
        self.y_train= y_train

    def TunedModel(self):
        SVMTune = Tuner(SVCParams, SVCModel, self.X_train, self.y_train)
        GridSearchRes = SVMTune.GridSearch()
        RandomSearchRes = SVMTune.RandomSearch()
        if GridSearchRes[0] > RandomSearchRes[0]:
            model = GridSearchRes[1].fit(self.X_train, self.y_train)
        else:
            model = RandomSearchRes[1].fit(self.X_train, self.y_train)
        return model

