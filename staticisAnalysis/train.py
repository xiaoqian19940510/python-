import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
%matplotlib inline
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
train = spark.sql("select * from mydb.t231_train1d")
pd_train = train.toPandas()

#模型训练
#pd_train1 = pd_train1.drop(['xd13','xd28','xd37','xd67','xd69','xd71','xd100','xd101','xd102','xd103','xd104','xd105','xd106','xd107','xd108','xd109','xd110','xd111','xd112','xd113','xd114','xd115','xd116','xd117','xd118','xd119','xd120','xd121','xd122','xd123','xd124','xd125','xd126','xd127','xd128','xd129','xd130','xd131','xd132','xd133','xd134'],axis=1)
pd_train = pd_train.drop(['xd13','xd28','xd37','xd67','xd69','xd71','xd100','xd101','xd102','xd103','xd104','xd105','xd106','xd107','xd108','xd109','xd110','xd111','xd112','xd113','xd114','xd115','xd116','xd117','xd118','xd119','xd120','xd121','xd122','xd123','xd124','xd125','xd126','xd127','xd128','xd129','xd130','xd131','xd132','xd133','xd134','xd148','xd149','xd150','xd151','xd152','xd153','xd154','xd155'],axis=1)
#pd_train = pd_train0.append(pd_train1)
pd_train = pd_train.dropna()
label1 = pd_train2['label']
pd_train2['label'] = label1.astype(float)
#y_label = pd_train['label']
X_train, X_test, Y_train, Y_test = train_test_split(pd_train2.drop(['label'],axis=1), pd_train2['label'], test_size = 0.3)
X_train['source'] = 'train'
X_test['source'] = 'test'
X_train.drop('source',axis=1,inplace=True)
X_test.drop('source',axis=1,inplace=True)
import numpy as np
import pandas as pd
from pandas import  DataFrame
from patsy import dmatrices
import string
from operator import itemgetter
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split,StratifiedShuffleSplit,StratifiedKFold
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.externals import joblib
seed= 0
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if string.find(big_string, substring) != -1:
            return substring
    print big_string
    return np.nan

X_train = X_train.drop(['eqid','wtid','wfid','logtime'],axis=1)
#print X_train
#Y_train = np.asarray(Y_train).ravel()
X_test = X_test.drop(['eqid','wtid','wfid','logtime'],axis=1)

clf = RandomForestClassifier(n_estimators=100,criterion='gini',n_jobs = -1,min_samples_leaf=60,max_features='auto',oob_score=True,random_state=seed,verbose=0)
param_grid = dict()
pipeline = Pipeline([('clf',clf)])
grid_search = GridSearchCV(pipeline,param_grid=param_grid,verbose=3,scoring='accuracy',cv=StratifiedShuffleSplit(Y_train,n_iter=2,test_size=0.2,train_size=None,random_state=seed)).fit(X_train,Y_train)
print("Best score: %0.3f" % grid_search.best_score_)
print(grid_search.best_estimator_)
print ('on all train set')
scores = cross_val_score(grid_search.best_estimator_, X_train, Y_train,cv=3,scoring='accuracy')
print("Best score: %0.3f" % grid_search.best_score_)
print(grid_search.best_estimator_)
print scores.mean(),scores
print ('on test set')
scores = cross_val_score(grid_search.best_estimator_, X_test, Y_test,cv=3,scoring='accuracy')
print scores.mean(),scores
print(classification_report(Y_train, grid_search.best_estimator_.predict(X_train) ))
print('test data')
print(classification_report(Y_test, grid_search.best_estimator_.predict(X_test) ))
aa = grid_search.best_estimator_.predict(X_test)
from sklearn.metrics import confusion_matrix  
confmat = confusion_matrix(Y_test,aa,labels=list(set(Y_test)))
model = grid_search.best_estimator_
model1 = clf.fit(X_train,Y_train)
impt = clf.feature_importances_
impd = pd.DataFrame(impt)
impd['impt1'] = impt
X_train = DataFrame(X_train)
#X_train.columns = pd_train.drop(['label','eqid','wtid','wfid','logtime'],axis=1).columns
model = grid_search.best_estimator_
model1 = clf.fit(X_train,Y_train)
impt = clf.feature_importances_
impd = pd.DataFrame(impt)
impd['impt1'] = impt
import matplotlib.pyplot as plt
fig = plt.figure()
fig.set_figheight(20)
fig.set_figwidth(20)
fig.set(alpha=0.2)  # 设定图表颜色alpha参数
impt2 = sorted(impd.impt1)
impd['var'] = X_train.columns
impd = impd.sort_values('impt1',ascending=False)
impd = impd[['impt1','var']]
impd1 = impd.head(20)
impd1.plot(kind='barh',x=impd1['var'],figsize=(10,30))
