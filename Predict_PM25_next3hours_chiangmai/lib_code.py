import numpy as np
import pandas as pd
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from selenium.webdriver.support.select import Select
import time
from bs4 import BeautifulSoup
from datetime import datetime, date,timedelta
import math
import pickle
import matplotlib.pyplot as plt
from os import mkdir
from os.path import isdir

     

def get_install_driver():
    driver = webdriver.Chrome(ChromeDriverManager().install())
    return driver

def gotopage(driver,target_url,station_index):
    driver.get(target_url)

    #select station
    station = Select(driver.find_element_by_css_selector('select[id="stationId"]'))
    station.select_by_value(station_index)

    param = Select(driver.find_element_by_id('parameterSelected'))

    time.sleep(10)

    html = driver.page_source
    soup = BeautifulSoup(html)

    # each staton has different parameter options
    select = soup.find_all(attrs={'id':'parameterSelected'})[0]
    len_options = len(select.find_all('option'))

    for i in range(len_options):
        param.select_by_index(i)

    element = driver.find_element_by_id("datepicker_start")
    string = "arguments[0].setAttribute('value',"+"'"+ str(date.today() - timedelta(0))+"')"
    driver.execute_script(string, element)

    element = driver.find_element_by_id("datepicker_end")
    string = "arguments[0].setAttribute('value',"+"'"+ str(date.today() - timedelta(0))+"')"
    driver.execute_script(string, element)
    
    # select time
    start_hr = Select(driver.find_element_by_id('startHour'))
    start_hr.select_by_index(0)
    start_min = Select(driver.find_element_by_id('startMin'))
    start_min.select_by_index(0)
    stop_hr = Select(driver.find_element_by_id('endHour'))
    stop_hr.select_by_index(23)
    stop_min = Select(driver.find_element_by_id('endMin'))
    stop_min.select_by_index(59)

    # Retrive data
    button = driver.find_element_by_name('bt_show_table')
    button.click()
    time.sleep(10)

def page_data(result_soup):
    table = result_soup.find_all(attrs = {'id':'table_mn_div'})[0]
    table = table.table.tbody
    head = table.find_all('tr')[0]
    head_text = [text for text in head.stripped_strings]
    #create feature name in dataframe
    matrix = []
    matrix = np.hstack(head_text)

    #create data in each row in dataframe
    body = table.find_all('tr')[1:]

    for row in body:
        data_s = row.find_all('input')
        # the last <input> tag is empty
        if len(data_s) != 0:
            row_data = [data['value'] for data in data_s]
            matrix = np.vstack((matrix, row_data))
            
    
    page_df = pd.DataFrame(matrix[1:,:], columns=matrix[0,:])
    return page_df


def save_page_data(pagesource,station_index):
    base_dir = 'Data/'
    type_1 = 'csv'
    type_2 = 'html'
    if not(isdir(base_dir+station_index)):
        mkdir(base_dir+station_index)
    if not(isdir(base_dir+station_index+'/'+type_2)):
        mkdir(base_dir+station_index+'/'+type_2)
    if not(isdir(base_dir+station_index+'/'+type_1)):
        mkdir(base_dir+station_index+'/'+type_1)

    filename_html = "Data/"+station_index+"/html/"+str(date.today()-timedelta(0))+'.html'
    filename_csv = "Data/"+station_index+"/csv/"+str(date.today()-timedelta(0))+'.csv'
    with open(filename_html,'w',encoding='utf-8') as f:
        f.write(pagesource)


    with open(filename_html,encoding='utf-8') as f:
        result_soup = BeautifulSoup(f.read())
    page_df = page_data(result_soup) #extract data from html
    df = pd.DataFrame()
    df = pd.concat([page_df])
    df['Date'] = df['Date'].apply(lambda x: datetime.strptime(x,'%Y,%m,%d,%H,%M,%S'))
    df.set_index(df['Date'],drop=True,inplace=True)
    df.drop('Date',axis=1,inplace=True)
    df.to_csv(filename_csv)

    df.drop(['35t_THC (ppm)','35t_CO (ppm)','35t_SRAD (w/m2)','35t_NRAD (w/m2)','35t_BP (mmHG)',"35t_RAIN (mm)",'35t_O3 (ppb)'],axis=1,inplace=True)
    df.columns = ['NO','NOX','NO2','SO2','PM10','Wind dir','Temp','Rel hum','Wind speed','PM2.5']
    list_col = ['NO','NOX','NO2','SO2','PM10','Wind dir','Temp','Rel hum','Wind speed','PM2.5']

    df = df.replace('-',np.nan)

    for col in df.columns:
        df[col] = df[col].astype(float)

    df[list_col] = df[list_col].interpolate(method='time',limit_direction='backward')
    df = df.fillna(0)
    return df

def resample(hour,day,df):
    df = df.iloc[:-1].copy().resample(hour,label='right',closed='right').agg({'NO':np.mean,'NOX':np.mean,'NO2':np.mean,'SO2':np.mean,
                                                   'PM10':np.mean,'Temp':np.mean,'Rel hum':np.mean,'Wind speed':np.mean,
                                                   'Wind dir':'last','PM2.5':np.mean})
    if day=='yesterday':
        df = df[str(date.today() - timedelta(1))]
    elif day == 'today':
        df = df[str(date.today() - timedelta(0))]
    return df

def create_cyclical_data(df,col,max_):
#     print(col)
#     print('max cyclic of df : ',max_)
    df['sin_'+col] = np.sin((df[col]*2*np.pi)/max_)
    df['cos_'+col] = np.cos((df[col]*2*np.pi)/max_)
    return df


def process_pcd_data(df,method):
    df['Wind speed'] = df['Wind speed'] * 1.609344 #change wind speed unit mph to kmph

    # Craete season
    df['season'] = 'other'
    df.loc['2020-02-16 00:00:00':'2020-05-05 23:00:00','season'] = 'summer'
    df.loc['2020-05-16 00:00:00':'2020-10-05 23:00:00','season'] = 'rainy'
    df.loc['2020-10-16 00:00:00':'2021-02-05 23:00:00','season'] = 'winter'
    
    df.loc['2020-02-06 00:00:00':'2020-02-15 23:00:00','season'] = 'winter-summer'
    df.loc['2020-05-06 00:00:00':'2020-05-15 23:00:00','season'] = 'summer-rainy'
    df.loc['2020-10-06 00:00:00':'2020-10-15 23:00:00','season'] = 'rainy-winter'
    
    df['daymonth'] = df.index.day
    df['dayname'] = df.index.strftime('%A')
    df['is_weekend'] = 'is_weekday'
    df.loc[df['dayname'].isin(['Saturday','Sunday']),'is_weekend'] = 'weekend'
    
#     #convert Wind dir to string for dummy dataset
#     df['Wd_string'] = df['Wind dir'].apply(degToCompass)
    
    df['location'] = 1 # if city_hall = 0
    # convert category data | cyclical data to numberic
    df = df.replace('is_weekday',0)
    df = df.replace('is_weekend',1)
    if method == 'cyclical':
        df['hour'] = df.index.hour
        df['month'] = df.index.month

        # day to numberic
        df = df.replace('Friday',0)
        df = df.replace('Monday',1)
        df = df.replace('Saturday',2)
        df = df.replace('Sunday',3)
        df = df.replace('Thursday',4)
        df = df.replace('Tuesday',5)
        df = df.replace('Wednesday',6)
        
         # season to numberic
        df = df.replace('rainy',1)
        df = df.replace('rainy-winter',2)
        df = df.replace('summer',3)
        df = df.replace('summer-rainy',4)
        df = df.replace('winter',5)
        df = df.replace('winter-summer',6)
        
        
        cyclical = create_cyclical_data(df.copy(),'Wind dir',360)
        cyclical.drop('Wind dir',axis=1,inplace=True)
        columns = ['hour','month','dayname','daymonth','season']
        max_list = [24,12,6,31,6]
        for i,col in enumerate(columns):
            cyclical = create_cyclical_data(cyclical.copy(),col,max_list[i])
            cyclical.drop(col,axis=1,inplace=True)
        return cyclical
    elif method == 'dummy':
#         df['daymonth'] = df.index.day.astype(str)
        df['hour'] = df.index.hour.astype(str)
        df['month'] = df.index.strftime('%B').astype(str)
        dummy = pd.concat([df.drop(['daymonth','month','dayname','is_weekend','hour','season'],axis=1),
                  pd.get_dummies(df[['daymonth','month','dayname','is_weekend','hour','season']])])
        
        return dummy

def adjust_lag(df):
    # make all data to lag1
    df.rename(columns={'PM2.5':'y'},inplace=True)
#     print(len(df))
    
    df2 = df[df['location'] == 1]
    
    df2['PM2.5'] = df2['y'].shift(1)

    df2[list(df2.drop(['PM2.5','y'],axis=1).columns)] = df2[list(df2.drop(['PM2.5','y'],axis=1).columns)].shift(1)
    
#     df = pd.concat([df1.dropna(),df2.dropna()])
    
#     print(len(df))
    return df2.dropna()

def process(dataframe):
    list_col = ['NO', 'NOX', 'NO2', 'SO2', 'PM10', 'Temp', 'Rel hum', 'Wind speed','PM2.5']
    #process data 
    dataframe.replace('-',np.nan)
    for col in dataframe.columns:
        dataframe[col] = dataframe[col].astype(float)

    #feature engineering
    dataframe[list_col] = dataframe[list_col].interpolate(method='time',limit_direction='backward')
    dataframe = dataframe.fillna(0)

        #->resample to 3 hour
    cyclical = resample('3H','today',dataframe)
    cyclical_lag = adjust_lag(cyclical.copy())

def merc_x(lon):
  r_major = 6378137.000 # unit:meter
  return r_major*math.radians(lon)

def merc_y(lat):
    lat += 0.08
    if lat>89.5:lat=89.5
    if lat<-89.5:lat=-89.5
    r_major=6378137.000
    r_minor=6356752.3142
    temp=r_minor/r_major
    eccent=math.sqrt(1-temp**2)
    phi=math.radians(lat)
    sinphi=math.sin(phi)
    con=eccent*sinphi
    com=eccent/2
    con=((1.0-con)/(1.0+con))**com
    ts=math.tan((math.pi/2-phi)/2)/con
    y=0-r_major*math.log(ts)
    return y

def handle_firms_data(df):
    df = df.sort_values(['acq_date','acq_time','latitude','longitude'])
    df = df.drop_duplicates()
    df.reset_index(drop=True,inplace=True)
    
    df['x_mer']= df.longitude.apply(merc_x)
    df['y_mer']= df.latitude.apply(merc_y)

    #calculate distance from chiang mai
    df['distance2chiang_mai'] = np.sqrt((df.x_mer - 11018321.156253504)**2 +(df.y_mer - 2126542.246966054)**2)

    #assemble datetime column \
    df['datetime'] = df['acq_date'] + ' ' + df['acq_time']
    df['datetime'] = pd.to_datetime(df['datetime'],format='%Y-%m-%d %H%M',utc=True)

    #convert to Bangkok time zone and remove time zone information
    df['datetime'] = df['datetime'].dt.tz_convert('Asia/Bangkok')
    df['datetime'] = df['datetime'].dt.tz_localize(None)
    df = df.sort_values('datetime')
    df.drop(['acq_time','acq_time','latitude','longitude'],axis=1,inplace=True)
    df.set_index('datetime',inplace=True)
    return df

def handle144(processed_firms):
    fireclose = processed_firms[processed_firms['distance2chiang_mai'].values<144000]
    firehourly144 = fireclose.resample('3H',label='right',closed='right').agg({'power':['sum','count']})
    firehourly144.columns = ['pwsum144','firecount144']

    firehourly144 = firehourly144.shift(2)
    # # create a 24 hour rolling sum 
    firehourlyroll144 = firehourly144.rolling(window=8).sum().dropna()
    return firehourlyroll144

def handle288(processed_firms):
    fireclose = processed_firms[(processed_firms['distance2chiang_mai'].values>144000)&(processed_firms['distance2chiang_mai'].values < 288000)]
    firehourly288 = fireclose.resample('3H',label='right',closed='right').agg({'power':['sum','count']})
    firehourly288.columns = ['pwsum288','firecount288']

    firehourly288 = firehourly288.shift(2+8)
    firehourlyroll288 = firehourly288.rolling(window=8).sum().dropna()
    return firehourlyroll288

def handle432(processed_firms):
    fireclose = processed_firms[(processed_firms['distance2chiang_mai'].values<432000)&(processed_firms['distance2chiang_mai'].values > 288000)]
    firehourly432 = fireclose.resample('3H',label='right',closed='right').agg({'power':['sum','count']})
    firehourly432.columns = ['pwsum432','firecount432']

    firehourly432 = firehourly432.shift(2+8+8)
    firehourlyroll432 = firehourly432.rolling(window=8).sum().dropna()
    return firehourlyroll432

def handleg432(processed_firms):
    fireclose = processed_firms[(processed_firms['distance2chiang_mai'].values<3000000)&(processed_firms['distance2chiang_mai'].values > 432000)]
    firehourlyg432 = fireclose.resample('3H',label='right',closed='right').agg({'power':['sum','count']})
    firehourlyg432.columns = ['pwsumg432','firecountg432']

    # shift the data by 8 hours คือความล่าช้าของมวล
    firehourlyg432 = firehourlyg432.shift(2+8+8+8+8+8+8)
    # create a 24 hour rolling sum 
    firehourlyrollg432 = firehourlyg432.rolling(window=3).sum().dropna()
    return firehourlyrollg432


def process_firm_data(dataframe):

    firms_df = dataframe[['latitude', 'longitude','brightness','acq_date','acq_time','power']]

    firms_df['latitude'] = firms_df['latitude'].astype(float)
    firms_df['longitude'] = firms_df['longitude'].astype(float)
    firms_df['brightness'] = firms_df['brightness'].astype(float)
    firms_df['power'] = firms_df['power'].astype(float)

    processed_firms = handle_firms_data(firms_df.copy())
    firehourlyroll144 = handle144(processed_firms)
    firehourlyroll288 = handle288(processed_firms)
    firehourlyroll432 = handle432(processed_firms)
    firehourlyrollg432 = handleg432(processed_firms)

    merge_list = [firehourlyroll144, firehourlyroll288, firehourlyroll432,firehourlyrollg432]
    new_fire = pd.DataFrame(index=firehourlyroll144.index)
    for df in merge_list:
        new_fire = new_fire.merge(df,left_index=True,
                                        right_index=True, how='outer' )
    new_fire.fillna(0,inplace=True)
    return new_fire


def merge_pcd_with_firms(pcd_df,firms_df):
    pcd_cf = pcd_df.merge(firms_df,left_index=True,right_index=True,how='inner')
    pcd_cf.fillna(0,inplace=True)
    return pcd_cf

def create_predict_df(df,model,x_test):
    times = pd.date_range(date.today()-timedelta(0), periods=len(df)+3, freq='3H')
    times = times[1:]
    new_df = pd.DataFrame(index=times)
    new_df['y'] = df['y']
    new_df['predict'] = np.nan
    new_df['predict'].iloc[1:-1] = model.predict(x_test)
    return new_df


def create_predict_graph(df,title):
    plt.figure(figsize=(9,8))
    df.plot(figsize=(9,8),legend=True,marker='o',title=title+str(date.today()-timedelta(0)))
    plt.text(df.index[-2],df.iloc[-2]['predict'],'%.2f'%(df.iloc[-2]['predict']),fontsize=15)
    plt.savefig("predict/"+title+str(date.today()-timedelta(0))+'.png')


def predict_data(df,type_,station_id,model):
    if type_ == "pcd_firms":
        cols_select = ['PM2.5','PM10','Rel hum','Temp','sin_hour','cos_Wind dir',
              'NO2','sin_Wind dir','Wind speed','sin_daymonth','cos_hour',
               'NO','NOX','cos_daymonth','sin_month','SO2','cos_month',
              'pwsum144', 'firecount144','pwsum288', 'firecount288', 'pwsum432',
               'firecount432','pwsumg432','firecountg432']


        x_pcd_cf = df.drop('y',axis=1)
        y_pcd_cf = df[['y']]
        x_pcd_cf = x_pcd_cf[cols_select]
        result_pcd_firms = create_predict_df(y_pcd_cf,model,x_pcd_cf)

        title = 'Predict PM2.5 Value with PCD FIRMS Dataset by Xgboost at %s '%(station_id)
        create_predict_graph(result_pcd_firms,title)


        return result_pcd_firms['predict'].iloc[-2]
    elif type_ == "pcd":
        cols_select = ['PM2.5','PM10','Rel hum','Temp','sin_hour','cos_Wind dir',
              'NO2','sin_Wind dir','Wind speed','sin_daymonth','cos_hour',
               'NO','NOX','cos_daymonth','sin_month','SO2','cos_month','y']

        cyclical_lag = df[cols_select]
        x_cyclical_lag = cyclical_lag.drop('y',axis=1)
        y_cyclical_lag = cyclical_lag[['y']]
        result_pcd = create_predict_df(y_cyclical_lag,model,x_cyclical_lag)


        title = 'Predict PM2.5 Value with PCD  Dataset by Xgboost at %s '%(station_id)
        create_predict_graph(result_pcd,title)

        return result_pcd['predict'].iloc[-2]


