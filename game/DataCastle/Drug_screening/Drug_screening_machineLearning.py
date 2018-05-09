
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn import ensemble
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn import svm
#data process
train=pd.read_csv('taijing/df_affinity_train.csv')
test=pd.read_csv('taijing/df_affinity_test_toBePredicted.csv')
train=pd.DataFrame(train)
test=pd.DataFrame(test)
test.columns = ['Protein_ID','Molecule_ID','Ki']
del test['Ki']
train.columns = ['Protein_ID','Molecule_ID','Ki']
train.dropna(inplace=True)
# del train['id']
# train = shuffle(train)
label=train['Ki']
del train['Ki']

min_max_scaler = preprocessing.MinMaxScaler()
train = min_max_scaler.fit_transform(train)
# train=preprocessing.normalize(train, norm='l2')
train=preprocessing.scale(train, axis=0, with_mean=True,with_std=True,copy=True)

from sklearn.cross_validation import train_test_split
train_data,test_data,train_label,test_label= train_test_split(train,label,test_size=0.3,
                                        random_state=1)

a,b,c,d,e,f=0.0,0.1,0.7,0.0,0.1,0.1
params = {'n_estimators': 1000}
# clf1 = ensemble.GradientBoostingRegressor(**params)
clf1= ensemble.RandomForestRegressor(n_estimators = 1000,criterion='mse',bootstrap=True,warm_start=False,min_weight_fraction_leaf=0.0, min_samples_split=3,
                                      random_state=2,n_jobs=-1,max_features = "auto",min_samples_leaf = 1)
# # clf4 = ensemble.AdaBoostRegressor(**params)
# clf5 = ensemble.BaggingRegressor(**params)
# clf6 = ensemble.ExtraTreesRegressor(**params)
# clf1 = svm.SVR(kernel='rbf',degree=3, gamma='auto', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)


clf1.fit(train_data, train_label)
# clf2.fit(train_data, train_label)
# clf3.fit(train_data, train_label)
# clf4.fit(train_data, train_label)
# clf5.fit(train_data, train_label)
# clf6.fit(train_data, train_label)
#训练
y_train_pred1 = clf1.predict(train_data)
# y_train_pred2 = clf2.predict(train_data)
# y_train_pred3 = clf3.predict(train_data)
# y_train_pred4 = clf4.predict(train_data)
# y_train_pred5 = clf5.predict(train_data)
# y_train_pred6 = clf6.predict(train_data)
y_train_pred=y_train_pred1
# y_train_pred=a*y_train_pred1+b*y_train_pred2+c*y_train_pred3+d*y_train_pred4+e*y_train_pred5+f*y_train_pred6
y_test_pred1 = clf1.predict(test_data)
# y_test_pred2 = clf2.predict(test_data)
# y_test_pred3 = clf3.predict(test_data)
# y_test_pred4 = clf4.predict(test_data)
# y_test_pred5 = clf5.predict(test_data)
# y_test_pred6 = clf6.predict(test_data)
# y_test_pred=a*y_test_pred1+b*y_test_pred2+c*y_test_pred3+d*y_test_pred4+e*y_test_pred5+f*y_test_pred6
y_test_pred=y_test_pred1
print('R^2 train: %.3f, test: %.3f' % (r2_score(train_label, y_train_pred),
                                   r2_score(test_label, y_test_pred)))

mse = mean_squared_error(test_label, y_test_pred)
#预测并且计算MSE #print 2.7和3.0版本有区别
print("MSE: %.4f" % mse)
print('END.....\n')
test_label=clf1.predict(test)
test.insert(0,'Ki',test_label)
test.to_csv('taijing/result_final.csv')
