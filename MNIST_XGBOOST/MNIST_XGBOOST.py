"""
MINIST XGBOOST¶
Business Understanding: Using MINIST dataset to train model , then recognize digit image
Analytic Approach: XGBOOST
Data Requirement: using minist data Kaggle digit recognizer
Data Collection: using pandas, read from csv file
Data Understanding: using pandas and visulizatio seaborn, it's easy, no complex featurs, only pixels(724)
Data Preparation: pandas , no need more effort
Modeling: xgboost, scikit-learn, GridSearchCV
Evaluation: scikit-learn, confusion matrix, F1 Score, ROC/AUC
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import time

#Data Collection
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#Data Understanding
print('Train shape is:',train.shape)
print('Test shape is:', test.shape)

#Data Preparation
X = train.iloc[:, 1:]
y = train.iloc[:, 0]

#Modeling
#Split train set and validation set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1)

print("X_train shape is:", X_train.shape)
print("X_val shape is:", X_val.shape)
print("y_train shape is:", y_train.shape)
print("y_val shape is:", y_val.shape)

#xgboost matrix creation
xgb_train = xgb.DMatrix(X_train, label=y_train)
xgb_val   = xgb.DMatrix(X_val, label=y_val)
xgb_test  = xgb.DMatrix(test)

#specify parameters
params={
'booster':'gbtree',
'objective': 'multi:softmax', #multi classification
'num_class':10, # classes number as this is multi classification
'gamma':0.1,  # prunching parameter, normally , 0.1, 0.2, larger , less prunching
'max_depth':12, # depth of tree, bigger tree will lead to overfitting
'lambda':2,  #  model complexity , L2 regularization, bigger -> less overfitting
'subsample':0.7, # sampling randomly
'colsample_bytree':0.7, # column sampling when tree creation
'min_child_weight':3, 

# ???
# 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
#，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
#这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。 
# smaller, easier overfitting
    
#Minimum sum of instance weight (hessian) needed in a child. 
#If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight,
#then the building process will give up further partitioning. 
#In linear regression task, this simply corresponds to minimum number of instances needed to be in each node. 
#The larger min_child_weight is, the more conservative the algorithm will be.    
'silent':0 ,# running infor , set to : 0.
'eta': 0.007, # like learning rate
'seed':1000,
'nthread':7,# cpu threads
'tree_method': 'gpu_hist' # Use GPU accelerated algorithm    
#'eval_metric': 'auc'
}

#specify validations set to watch performance
watchlist = [(xgb_train, 'train'), (xgb_val, 'eval')]

#num_iteration = 1200
num_iteration = 1

#Train
model = xgb.train(params, xgb_train, num_boost_round= num_iteration, evals=watchlist, early_stopping_rounds=100)

model.save_model('minist_xgboost.model')
print('Best best_ntree_limit', model.best_ntree_limit)

#Evaluation

# Can not predict , flash
bst2 = xgb.Booster(model_file='minist_xgboost.model.backup')
preds = bst2.predict(xgb_test, ntree_limit=1169)

preds_df = pd.DataFrame(preds)
preds_df['ImageId'] = preds_df.index
preds_df['Label']=preds_df.iloc[:,0].astype(int)
preds_df.drop(0, axis=1, inplace=True)
preds_df.to_csv('xgb_submission3.csv', index=False)

#np.savetxt('xgb_submission.csv',np.c_[range(1,len(test)+1),preds],delimiter=',',header='ImageId,Label',comments='',fmt='%d')
#np.savetxt('xgb_submission.csv',preds,delimiter=',',header='ImageId,Label',comments='',fmt='%d')