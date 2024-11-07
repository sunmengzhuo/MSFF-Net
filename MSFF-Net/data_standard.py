import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 定义函数去除非数值型特征
def getNum(data,data_new):
    columns = data.columns
    for col in columns:
        try:
            df = data[col].astype(np.float64)
            data_new = pd.concat([data_new, df], axis=1)
        except:
            pass
        continue
    return data_new


dataPath = "D:\\结直肠癌\\腹膜转移\\2y\\2y.xlsx"
data = pd.read_excel(dataPath)
dataNew = pd.DataFrame()
dataNew = getNum(data, dataNew)
dataNew.index = range(len(dataNew))
X = dataNew[dataNew.columns[2:]]
y = dataNew['label']
Z = dataNew['num']

# 将数据类型转换为数值型
X = X.apply(pd.to_numeric, errors='ignore')
colNames = X.columns
X = X.fillna(0)
X = X.astype(np.float64)
# 标准化处理
X = StandardScaler().fit_transform(X)
X = pd.DataFrame(X)
X.columns = colNames

data_ = pd.concat([Z, y, X], axis=1)
data_.to_csv("./feature_standard_ex_test.csv",encoding='gb18030')

