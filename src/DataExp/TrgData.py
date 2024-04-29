from sklearn.preprocessing import MinMaxScaler

from src.DataExp.DataSource import *
from sklearn.model_selection import train_test_split


class DataPreparation:
    def __init__(self, OrgData):
        self.OrgData= OrgData

    def DataStat(self):
        print(f'The initial data has {self.OrgData.shape[0]} feature vectors with {self.OrgData.shape[1]} features\n')
        a = self.OrgData['Out'].sum(axis=0)
        print(f'Number of positive cases in the data is: {a}\n')

    def split_data(self,SplitRatio,scaler):
        print('Preparing the training data...')
        Data= self.OrgData
        X = pd.DataFrame(Data.iloc[:, :-1])
        y = pd.DataFrame(Data.iloc[:, -1])
        X = scaler.fit_transform(X)
        X = pd.DataFrame(X, index=Data.index, columns=Data.iloc[:, :-1].columns)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=SplitRatio)
        print(f'After Split, Length of training data vectors is {X_train.shape[0]} and test data vectors is {X_test.shape[0]}\n')
        y_train= y_train.values.ravel()
        #y_test= y_test.values.ravel()
        return X_train, X_test, y_train, y_test
