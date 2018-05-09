#Fault classification and detection of charging pile
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
train=pd.read_csv('data_train.csv')
test=pd.read_csv('data_test.csv')
train.columns = ['id','a','b','c','d','e','f','label']
test.columns = ['id','a','b','c','d','e','f','label']
train=pd.DataFrame(train)
test=pd.DataFrame(test)

del train['id']
train = shuffle(train)
label=train['label']
del train['label']
length=len(train)
key=int(0.8*length)
train_data=train[:key]
train_label=label[:key]
val_data=train[key:]
val_label=label[key:]
# print(train.head(6))
data_train = xgb.DMatrix(train_data, label=train_label)
data_val = xgb.DMatrix(val_data, label=val_label)
watch_list = [(data_val, 'eval'), (data_train, 'train')]

#train
param = {'n_estimators': 10,'max_depth': 10, 'eta': 0.8, 'silent': 1, 'objective': 'multi:softmax', 'num_class': 2,'subsample': 0.8,'colsample_bytree':0.7,'min_child_weight':3,'seed':1000,'booster':'gbtree'}
bst = xgb.train(param, data_train, num_boost_round=5000, evals=watch_list)
bst.save_model('Fault_classify/0001.model')
# val accuracy
y_hat = bst.predict(data_val)
result = val_label.reshape(1, -1) == y_hat
print('accuracy is :\t', float(np.sum(result)) / len(y_hat))
print('END.....\n')

#predict

test['label']=0
del test['id']
test_label=test['label']
del test['label']
data_test = xgb.DMatrix(test, label=test_label)
y_pre = bst.predict(data_test)
test.insert(6,'label',y_pre)
length=len(test)
test.columns=['','','','','','','']
test.insert(0,'id',range(1,length+1))
# test.drop(0,axis=0)
test.to_csv('Fault_classify/result_finally.csv')




