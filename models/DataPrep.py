from src import DataExp
from src.DataExp import DataSource
from src.DataExp.DataSource import SelectData
from src.DataExp.TrgData import DataPreparation
from sklearn.preprocessing import MinMaxScaler, StandardScaler


Comp = SelectData('Metabollic')
Data= Comp.SelectFile()
TD= DataPreparation(Data)
TD.DataStat()
X_train, X_test, y_train, y_test = TD.split_data(0.3, MinMaxScaler())


