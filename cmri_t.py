import pandas as pd
from interval import IntervalSet
import time
import lightgbm as lgb
from dateutil.parser import parse
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from utils import *
pd.set_option('display.width',None)

data_path='./data_1.xlsx'
monend=2                #month end and month start
monstart=2
local_test=date(2018,9,15)
firstday=date(2017,1,1)
raw_data=pd.read_excel(data_path)
raw_data.columns=['timestamp','city','nettraffic']
raw_data=raw_data.reset_index(drop=True)
# print(raw_data.loc[raw_data['nettraffic'].isnull()])
# raw_data=raw_data.dropna()                              #drop 105 na records
raw_data=neigbh_fill(raw_data)                            #fill nan with neighb
test_data=testset('2018/11/16','2019/2/19')
raw_data=pd.concat([raw_data,test_data], axis=0)
raw_data=raw_data.reset_index(drop=True)

#basic feature
raw_data['date']=raw_data['timestamp'].apply(lambda x: x.date())
raw_data['holiday']=raw_data['timestamp'].apply(lambda x: holiday_dict(x.date()))
raw_data['city']=raw_data['city'].apply(lambda x: ord(x)-ord('A'))
raw_data['data_y']=raw_data['timestamp'].dt.year       #timestamp
raw_data['data_m']=raw_data['timestamp'].dt.month
raw_data['data_d']=raw_data['timestamp'].dt.day
raw_data['week']=raw_data['timestamp'].dt.dayofweek+1
# raw_data['weekend']=raw_data['week'].apply(lambda x: 1 if x>5 else 0)
raw_data['days']=raw_data['timestamp'].dt.days_in_month
raw_data['mon_end']=raw_data['days']-raw_data['data_d']
raw_data['mon_end']=raw_data['mon_end'].apply(lambda x: x if x<monend else 0) #last week of months
raw_data['mon_start']=raw_data['data_d']
raw_data['mon_start']=raw_data['mon_start'].apply(lambda x: x if x<monstart else 0)
raw_data['hour']=raw_data['timestamp'].dt.hour
raw_data['week_begin']=raw_data['date'].apply(lambda x: int(((x-firstday).days)/7))
# raw_data = lagging(raw_data, 7, 'nettraffic')
raw_data=laggingmean(raw_data, 10, 'nettraffic')
raw_data = laggingmean(raw_data, 20, 'nettraffic')
# raw_data['trend'] = raw_data['mean20lagging'] - raw_data['mean10lagging'] * 2

# temp=raw_data.groupby(['city','data_y','mon_start','hour'], as_index=False)['nettraffic'].agg({'mons_max':'max','mons_min':'min','mons_mean':'mean'})
# raw_data=raw_data.merge(temp,on=['city','data_y','mon_start','hour'],how='left')
# temp=raw_data.groupby(['city','data_y','mon_end','hour'], as_index=False)['nettraffic'].agg({'mone_max':'max','mone_min':'min','mone_mean':'mean'})
# raw_data=raw_data.merge(temp,on=['city','data_y','mon_end','hour'],how='left')
temp=raw_data.groupby(['data_y','city','data_d','hour'], as_index=False)['nettraffic'].agg({'mon_max':'max','mon_min':'min','mon_mean':'mean'})
temp=fill_periodic_fea(temp,['mon_max','mon_min','mon_mean'])
raw_data=raw_data.merge(temp,on=['data_y','city','data_d','hour'],how='left')
temp=raw_data.groupby(['data_y','city','week','hour'], as_index=False)['nettraffic'].agg({'w_max':'max','w_min':'min','w_mean':'mean'})
temp=fill_periodic_fea(temp,['w_max','w_min','w_mean'])
raw_data=raw_data.merge(temp,on=['data_y','city','week','hour'],how='left')

print(raw_data)
features=list(raw_data.columns)
features.remove('date')
features.remove('timestamp')
features.remove('nettraffic')
features.remove('data_y')
# features.remove('mean20lagging')
# features.remove('pre_nettraffic')
print(features,len(features))

#split testset
train=raw_data.loc[raw_data['nettraffic'].notnull()]
train_fea=train[features].values
train_y=train['nettraffic']
test=raw_data.loc[raw_data['nettraffic'].isnull()]
test_fea=test[features].values
result=test.copy()
test_index=test.index.tolist()
test_num=int(len(test_index)/24)


#lightGBM model
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mape',
    'num_leaves': 64,
    'min_data_in_leaf':10,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_seed':0,
    'bagging_freq': 1,
    'verbose': 1,
    'reg_alpha':1.7,
    'reg_lambda':4.9
}
lgb_train = lgb.Dataset(train_fea, train_y)
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=5000,
                valid_sets=[lgb_train],
                valid_names=['train','valid'],
                early_stopping_rounds=50,
                verbose_eval=500,
                )

res=np.array([])
for i in range(test_num):
    # print('\n')
    # print(test_fea[24*i:24*(i+1),-1:])
    res_sub = gbm.predict(test_fea[24*i:24*(i+1)])
    raw_data.loc[test_index[24*i:24*(i+1)], 'nettraffic'] = res_sub
    raw_data = laggingmean(raw_data, 10, 'nettraffic')
    raw_data = laggingmean(raw_data, 20, 'nettraffic')
    # raw_data['trend']=raw_data['mean20lagging']-raw_data['mean10lagging']*2
    test_fea=raw_data.loc[test_index][features].values
    res=np.hstack((res,res_sub))
print(raw_data)
result['nettraffic']=res
result['city']=result['city'].apply(lambda x: chr(ord('A')+int(x)))
result['nettraffic']=result['nettraffic'].apply(lambda x:('%.3f')%x)
result=result.ix[:,['timestamp','city','nettraffic']]
result.columns=['时间','地市','流量']
print(result)
result.to_excel("result.xlsx",index=False)



