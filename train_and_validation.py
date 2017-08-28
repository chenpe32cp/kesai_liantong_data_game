# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 10:52:43 2017

@author: PengChen
"""

import pandas as pd
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import lightgbm as lgb
from pandas import Series,DataFrame
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
#from sklearn import preprocessing

def load_raw_data():
    train_data = pd.read_csv(r"C:\competition\liantong\第1题：算法题数据\data\raw_data\数据集1_用户标签_本地_训练集.csv",encoding='utf-8',index_col='用户标识')
    test_data = pd.read_csv(r"C:\competition\liantong\第1题：算法题数据\data\raw_data\数据集1_用户标签_本地_测试集.csv",encoding='utf-8',index_col='用户标识')
    train_label=pd.read_csv(r"C:\competition\liantong\diyiti\data\raw_data\train_label.csv",index_col='用户标识')
    train_label = train_label.loc[train_data.index,:]
    return  train_data, test_data, train_label

def load_process_data(test_data=None):
    process_train_data = pd.read_csv(r"C:\competition\liantong\第1题：算法题数据\data\process_data\数据集1_用户标签_本地_训练集_process.csv",encoding='utf-8',index_col='用户标识')
    train_label=pd.read_csv(r"C:\competition\liantong\第1题：算法题数据\data\raw_data\数据集2_用户是否去过迪士尼_训练集.csv",index_col='用户标识')
    if test_data == True:        
        process_test_data = pd.read_csv(r"C:\competition\liantong\第1题：算法题数据\data\process_data\数据集1_用户标签_本地_测试集_process.csv",encoding='utf-8',index_col='用户标识')
        return process_train_data,process_test_data, train_label
    return process_train_data, train_label

#def feature_standard(process_train_data,process_test_data,train_label):
def feature_standard(process_train_data,train_label):
    #特征标准化
    #scaler = preprocessing.StandardScaler()
    #process2_data = scaler.fit_transform(process_train_data)
    #standard_test_data = scaler.transform(process_test_data)
    x_train, x_test, y_train, y_test = train_test_split(process_train_data,train_label,test_size=0.2,random_state=0)
    return x_train, x_test, y_train, y_test
    #return x_train, x_test, y_train, y_test, standard_test_data
    
#del process1_data, train_label
#gc.collect()

#使用xgboost的初始化的参数进行模型训练，注意参数未调优
def xgb_fit(x_train,y_train,x_test,y_test):
    #the second method
    xg_train = xgb.DMatrix(x_train, label=y_train,missing=0.0)
    xg_test = xgb.DMatrix(x_test, label=y_test,missing=0.0)
    params={'booster':'gbtree',
    	    'objective': 'binary:logistic',
    	    'eval_metric':'auc',
    	    'gamma':0.02,
    	    'min_child_weight':0.7,
    	    'max_depth':6,
    	    'lambda':2,
    	    'subsample':0.7,
    	    'colsample_bytree':0.7,
    	    'colsample_bylevel':0.7,
    	    'eta': 0.01,
    	    'tree_method':'exact',
    	    'seed':0,
    	    'nthread':12
    	    }
    watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
    model = xgb.train(params, xg_train, num_boost_round=5000, evals=watchlist,early_stopping_rounds=150)
    model.save_model('xgb.model') # 用于存储训练出的模型
#    print ("best best_ntree_limit",model.best_ntree_limit)
#    print ("跑到这里了model.predict")
#    preds = model.predict(xg_test,ntree_limit=model.best_ntree_limit)
#    test_recall=recall_score(y_test,preds)
#    print(test_recall)


#使用xgboost进行交叉训练，得到决策树数目。   
def xgb_modelfit(alg, x_train,y_train,x_test,y_test,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xg_train = xgb.DMatrix(x_train, label=y_train)
        cvresult = xgb.cv(xgb_param, xg_train, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics=['auc'], early_stopping_rounds=early_stopping_rounds, show_progress=True)
        alg.set_params(n_estimators=cvresult.shape[0]) 
    
    #Fit the algorithm on the data
    alg.fit(x_train, y_train.values,eval_metric='auc')
        
    #Predict training set:
    train_predictions = alg.predict(x_train)
    train_predprob = alg.predict_proba(x_train)[:,1]
        
    #Print model report:
    print ("\nModel Report")
    print ("Accuracy : %.4g" % metrics.accuracy_score(y_train.values, train_predictions))
    print ("AUC Score (Train): %f" % metrics.roc_auc_score(y_train.values, train_predprob))
    # Predict on testing data:  
    test_predictions=alg.predict_proba(x_test)[:,1] 
    print ('AUC Score (Test): %f' % metrics.roc_auc_score(y_test.values, test_predictions))
    
def tune_xgb_parameters(x_train,y_train,x_test,y_test):  
    xgb1 = XGBClassifier(learning_rate=0.01,n_estimators=1000,max_depth=6,min_child_weight=2,gamma=0,subsample=0.8,  
                        colsample_bytree=0.8,objective= 'binary:logistic',nthread=4,scale_pos_weight=1,seed=27)  
    xgb_modelfit(xgb1,x_train,y_train,x_test,y_test)



#使用lightgbm进行模型训练
def lgb_fit(x_train,y_train,x_test,y_test):

    lgb_train = lgb.Dataset(x_train,y_train['是否去过迪士尼'])
    lgb_test = lgb.Dataset(x_test,y_test['是否去过迪士尼'])
    params={
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'min_child_weight': 2,
    'num_leaves': 2 ** 8,
    'lambda_l2': 20,
    'subsample': 0.8,
    'feature_fraction':0.8,
    'bagging_fraction':0.8,
    'learning_rate': 0.01,
    'seed': 2017,
    'nthread': 12,
    	    }
    print('Start training...')
    bst = lgb.train(params,
                    lgb_train,
                    num_boost_round=5000,
                    valid_sets=[lgb_train,lgb_test],
                    #valid_sets=lgb_test,
                    valid_names=['train_data','test_data'],
                    early_stopping_rounds=150,
                    categorical_feature=['性别','手机品牌','手机终端型号','手机信息']
                    )   
    bst.save_model('lgb1.txt')
    return bst
    
def lgb_cv_fit(process2_data,train_label):
    lgb_train = lgb.Dataset(process2_data,train_label['是否去过迪士尼'])
    params={
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'min_child_weight': 2,
    'num_leaves': 2 ** 6,
    'lambda_l2': 20,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'colsample_bylevel': 0.7,
    'learning_rate': 0.01,
    'tree_method': 'exact',
    'seed': 2017,
    'nthread': 12,
    'silent': True
    	    }
    print('Start cv training...')
    bst = lgb.cv(params,
                    lgb_train,
                    num_boost_round=5000,
                    early_stopping_rounds=100,
                    nfold=5,
                    verbose_eval=True,
                    )   
    bst.save_model('lgb1.txt', num_iteration=bst.best_iteration)
    return bst   
 
def result(process_test_data, ypred):
    pred = DataFrame(ypred)
    a = process_test_data.reset_index()     
    b = pd.concat([a,pred],axis=1)
    final = b[['用户标识',0]]
    final.columns = ['IMEI','SCORE']
    final.to_csv('result.csv',index=False)

"""   
if __name__ == '__main__':
    #process_train_data,process_test_data, 
    train_label = load_process_data()
    x_train, x_test, y_train, y_test=feature_standard(process_train_data,train_label)
    bst=lgb_fit(x_train,y_train,x_test,y_test)    
    #bst=lgb_cv_fit(process2_data,train_label)
    ypred = bst.predict(standard_test_data, num_iteration=bst.best_iteration)
    result(process_test_data, ypred)
"""
"""   
    param_test1 = {
    'max_depth':list(range(3,10,2)),
    'min_child_weight':list(range(1,6,2))
    }
    gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.01, n_estimators=1000, max_depth=5,
     min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
     objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
     param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
    gsearch1.fit(x_train,y_train['是否去过迪士尼'])
    gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
    
                       
"""