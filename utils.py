from interval import Interval,IntervalSet
from dateutil.parser import parse
from datetime import date
import time
import pandas as pd
from workalendar.asia import China
from sklearn.metrics import mean_absolute_error

# china_holiday=China()
# day=parse("2019/2/21  0:00:00").date()
# print(day)
# print(china_holiday.is_working_day(day))

def holiday_dict(day):        #holiday dict
    china_holidays={date(2017,1,1):1,date(2017,1,2):1,date(2017,12,30):1,date(2017,12,31):1,date(2018,1,1):1,date(2018,12,30):1,date(2018,12,31):1,date(2019,1,1):1,
                    date(2017, 4, 2): 2, date(2017, 4, 3): 2, date(2017, 4, 4): 2,date(2018, 4, 5): 2,date(2018, 4, 6): 2,date(2018, 4, 7): 2,date(2019, 4, 5): 2,date(2019, 4, 6): 2,date(2019, 4, 7): 2,
                    date(2017, 5, 28): 3,date(2017, 5, 29): 3,date(2017, 5, 30): 3,date(2018, 6, 16): 3,date(2018, 6, 17): 3,date(2018, 6, 18): 3,date(2019, 6, 7): 3,date(2019, 6, 8): 3,date(2019, 6, 9): 3,
                    date(2018, 9, 22): 4,date(2018, 9, 23): 4,date(2018, 9, 24): 4,date(2019, 9, 13): 4,date(2019, 9, 14): 4,date(2019, 9, 15): 4,
                    date(2017, 1, 27): 5,date(2017, 1, 28): 5,date(2017, 1, 29): 5,date(2017, 1, 30): 5,date(2017, 1, 31): 5,date(2017, 2, 1): 5,date(2017, 2, 2): 5,}
    for i in range(7):
        china_holidays[date(2018,2,i+15)]=5
        china_holidays[date(2019, 2, i + 4)] = 5
        china_holidays[date(2017, 10, i + 1)] = 6
        china_holidays[date(2018, 10, i + 1)] = 6
        china_holidays[date(2019, 10, i + 1)] = 6
    if day in china_holidays:
        return china_holidays[day]
    else:
        return 0

def testset(start,end):
    cityname=['A','B','C']
    dt=pd.date_range(start, end, freq='H')
    result=pd.DataFrame(columns=['timestamp','city','nettraffic'])
    for city in cityname:
        cityrecord = pd.DataFrame()
        cityrecord['timestamp']=dt
        cityrecord['city']=city
        cityrecord=cityrecord[:-1]
        result = pd.concat([result, cityrecord], axis=0)
        result=result.ix[:,['timestamp','city','nettraffic']]
    return result

def neigbh_fill(raw_data):
    hour_f = raw_data['nettraffic'].shift(-1).fillna(method='ffill')
    hour_b = raw_data['nettraffic'].shift(1).fillna(method='bfill')
    temp = (hour_f + hour_b) / 2
    raw_data.loc[raw_data['nettraffic'].isna(), 'nettraffic'] = temp
    return raw_data

def fill_periodic_fea(data,features): #补全19年缺失的周期特征
    num=int(len(data)/3)
    for fea in features:
        increase = data[num:num*2][fea].values + data[num:num*2][fea].values - data[:num][fea].values
        data.loc[num*2:, fea] = increase
    return data

def lagging(data,day,origcol):
    colname='lagging'+str(day)
    num=day*24
    temp_lagging = pd.Series()
    for i in range(3):
        lagging = data[data['city']==i][origcol].shift(num)
        temp_lagging = pd.concat([temp_lagging, lagging])
    data[colname]=temp_lagging
    data[colname] = data[colname].fillna(data[origcol])
    return data

def laggingmean(data,days,origcol):
    colname='mean'+str(days)+'lagging'
    laggingm=pd.Series()
    for day in range(days):
        num=(day+1)*24
        temp_lagging = pd.Series()
        for i in range(3):
            lagging = data[data['city']==i][origcol].shift(num)
            temp_lagging = pd.concat([temp_lagging, lagging])
        temp_lagging=temp_lagging.fillna(data[origcol])
        if day:
            laggingm=laggingm+temp_lagging
        else:
            laggingm=temp_lagging
    data[colname]=laggingm
    return data

def laggmon(data,firstyear):
    data['datemon']=data['data_y']*100+data['data_m']
    data['datelaggmon']=data['datemon'].apply(lambda x: x-1 if (x%100)>1 else (x if int(x/100)==firstyear else x-89))
    if('laggmon' in list(data.columns)):
        data.drop(['laggmon'], axis=1, inplace=True)
    temp = data.groupby(['datemon','hour'], as_index=False)['nettraffic'].agg({'laggmon':'mean'})
    data = data.merge(temp, left_on=['datelaggmon','hour'], right_on=['datemon','hour'], how='left')
    data.drop(['datemon_x','datemon_y','datelaggmon'],axis=1,inplace=True)
    return data

def laggweek(raw_data):
    raw_data['wb_lagg']=raw_data['week_begin'].apply(lambda x:x-1 if x>0 else 0)
    if ('wb_max' in list(raw_data.columns)):
        raw_data.drop(['wb_max','wb_min','wb_mean'], axis=1, inplace=True)
    temp = raw_data.groupby(['week_begin', 'hour'], as_index=False)['nettraffic'].agg(
        {'wb_max': 'max', 'wb_min': 'min', 'wb_mean': 'mean'})
    raw_data = raw_data.merge(temp, left_on=['wb_lagg', 'hour'],right_on=['week_begin', 'hour'], how='left')
    raw_data['week_begin']= raw_data['week_begin_x']
    raw_data.drop(['week_begin_x','week_begin_y','wb_lagg'],axis=1,inplace=True)
    return raw_data



