import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
train=pd.read_csv('data/data_train1.csv',delimiter="\t",encoding='utf-8')
test=pd.read_csv('data/data_test1.csv',delimiter="\t",encoding='utf-8')
train=pd.DataFrame(train)
test=pd.DataFrame(test)
train.columns = ['id','a','b','label']
# del train['id']
ham1=train
# del ham1[]
# ham1.to_csv('ham_5000.utf8')
# ham2.insert(train[:100])
# ham2.to_csv('ham_5000.utf8')
# spam1.insert(train[:100])
# spam1.to_csv('spam_5000.utf8')
# spam2.insert(train[:100])
# spam2.to_csv('spam_5000.utf8')
