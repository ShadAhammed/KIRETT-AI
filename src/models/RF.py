import warnings
warnings.filterwarnings('ignore')
from src.models.ModelTuning import *
from models.ModelParams import *


class RandomForest:
    def __init__(self, X_train, y_train):
        self.X_train= X_train
        self.y_train= y_train

    def TunedModel(self):
        RFTune = Tuner(RFParams, RFModel, self.X_train, self.y_train)
        GridSearchRes = RFTune.GridSearch()
        RandomSearchRes = RFTune.RandomSearch()
        if GridSearchRes[0] > RandomSearchRes[0]:
            model = GridSearchRes[1].fit(self.X_train, self.y_train)
        else:
            model = RandomSearchRes[1].fit(self.X_train, self.y_train)
        return model

