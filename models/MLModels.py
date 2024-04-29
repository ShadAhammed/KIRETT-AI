from pip._internal.utils.misc import tabulate

from src.models.ANN import *
from tensorflow.keras.models import save_model, load_model
import numpy as np
from src.models.ModelTuning import Tuner
from ModelParams import *
from sklearn import metrics
# from DataPrep import *
from src.models.SVM import *
from src.models.XGB import *
from src.models.RF import RandomForest
from src.models.KNN import *
from src.models.NB import *
from src.models.LR import *
from src.DataExp.DataSource import SelectData
from src.DataExp.TrgData import DataPreparation
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
from sklearn import metrics
from src.visualization.Performance import *

Perf = pd.DataFrame()
# Preparing Data
Comp = SelectData('Metabollic')
Data = Comp.SelectFile()
TD = DataPreparation(Data)
TD.DataStat()
X_train, X_test, y_train, y_test = TD.split_data(0.3, MinMaxScaler())

# ANN model
model = 'ANN'
print(f'\n*********************** {model} ***********************\n')
InstANN = ANN(X_train, y_train)
ANNModel = InstANN.ANN_model(2)
prediction = np.round(ANNModel.predict(X_test))
p = ModelPerformance(X_test, y_test)
matrix_ANN = metrics.confusion_matrix(y_test, prediction)
PerformanceReport = p.ClfReport(y_test, prediction, model)
Perf = Perf.append(PerformanceReport)
p.draw_confusion_matrix(matrix_ANN)

# SVM Model
model = 'Support Vector'
print(f'\n*********************** {model} ***********************\n')
SVM = SupportVector(X_train, y_train)
SVModel = SVM.TunedModel()
p = ModelPerformance(X_test, y_test)
prediction = SVModel.predict(X_test)
matrix_SVM = metrics.confusion_matrix(y_test, prediction)
PerformanceReport = p.ClfReport(y_test, prediction, model)
Perf = Perf.append(PerformanceReport)
p.draw_confusion_matrix(matrix_SVM)
#
# XGB Model
model = 'XGB'
print(f'\n*********************** {model} ***********************\n')
XGB = XGB(X_train, y_train)
XGBModel = XGB.TunedModel()
p = ModelPerformance(X_test, y_test)
prediction = XGBModel.predict(X_test)
matrix_xgb = metrics.confusion_matrix(y_test, prediction)
PerformanceReport = p.ClfReport(y_test, prediction, model)
Perf = Perf.append(PerformanceReport)
p.draw_confusion_matrix(matrix_xgb)

#
# Random Forest Model
model= 'Random Forest'
RF = RandomForest(X_train, y_train)
print(f'\n*********************** {model} ***********************\n')
model = 'Random Forest'
RFModel = RF.TunedModel()
p = ModelPerformance(X_test, y_test)
prediction = RFModel.predict(X_test)
matrix_RF = metrics.confusion_matrix(y_test, RFModel.predict(X_test))
PerformanceReport = p.ClfReport(y_test, prediction, model)
Perf = Perf.append(PerformanceReport)
p.draw_confusion_matrix(matrix_RF)

#
# Logistic Regression Model
model = 'Logistic Regression'
print(f'\n*********************** {model} ***********************\n')
LR = LogRegression(X_train, y_train)
LRModel = LR.TunedModel()
p = ModelPerformance(X_test, y_test)
prediction = LRModel.predict(X_test)
matrix_LR = metrics.confusion_matrix(y_test, prediction)
PerformanceReport = p.ClfReport(y_test, prediction, model)
Perf = Perf.append(PerformanceReport)
p.draw_confusion_matrix(matrix_LR)

# Naive Bayes Model
model = 'Naive Bayes'
print(f'\n*********************** {model} ***********************\n')
NB = NaiveBayes(X_train, y_train)
NBModel = NB.TunedModel()
p = ModelPerformance(X_test, y_test)
prediction = NBModel.predict(X_test)
matrix_NB = metrics.confusion_matrix(y_test, prediction)
PerformanceReport = p.ClfReport(y_test, prediction, model)
Perf = Perf.append(PerformanceReport)
p.draw_confusion_matrix(matrix_NB)

# K-Nearest Neighbour Model
model = 'K-nearest Neighbour'
print(f'\n*********************** {model} ***********************\n')
KNN = KNearestNeighbour(X_train, y_train)
KNNModel = KNN.TunedModel()
p = ModelPerformance(X_test, y_test)
prediction = KNNModel.predict(X_test)
matrix_KNN = metrics.confusion_matrix(y_test, prediction)
PerformanceReport = p.ClfReport(y_test, prediction, model)
Perf = Perf.append(PerformanceReport)
p.draw_confusion_matrix(matrix_KNN)

print(f'\n****************************** Performance Report ******************************')
print(Perf)
