import warnings
warnings.filterwarnings('ignore')
from src.models.ModelTuning import *
from models.ModelParams import *


class NaiveBayes:

    def __init__(self, X_train,y_train):
        self.X_train= X_train
        self.y_train= y_train

    def TunedModel(self):
        NBTune = Tuner(NBParams, NBModel, self.X_train, self.y_train)
        GridSearchRes = NBTune.GridSearch()
        RandomSearchRes = NBTune.RandomSearch()
        if GridSearchRes[0] > RandomSearchRes[0]:
            model = GridSearchRes[1].fit(self.X_train, self.y_train)
        else:
            model = RandomSearchRes[1].fit(self.X_train, self.y_train)
        return model

