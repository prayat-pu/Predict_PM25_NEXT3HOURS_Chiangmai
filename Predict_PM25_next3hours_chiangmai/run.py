from lib_code import predict_data,merge_pcd_with_firms,get_install_driver,gotopage,save_page_data,resample,process_pcd_data,adjust_lag,process_firm_data
import pickle
from datetime import datetime
import pandas as pd


# web scrapping pcd data wevsite 
driver = get_install_driver()
target_url = 'http://www.aqmthai.com/public_report.php'
station_id = '35t'
gotopage(driver,target_url,station_id)
df = save_page_data(driver.page_source,station_id) #get dataframe

#process about data engineering
df = resample('3H','today',df)
cyclical = process_pcd_data(df.copy(),'cyclical')
cyclical_lag = adjust_lag(cyclical.copy())

# if you have data already you can skip line 21-23 but you have to select column ['latitude', 'longitude','brightness','acq_date','acq_time','power'] and send it to process_firm_data function
# link for download this file -> https://firms.modaps.eosdis.nasa.gov/data/active_fire/noaa-20-viirs-c2/csv/J1_VIIRS_C2_SouthEast_Asia_7d.csv
# use for test this file
filename = 'Data/test_firm_data/J1_VIIRS_C2_SouthEast_Asia_7d.csv'
firms_df = pd.read_csv(filename,usecols=[0,1,2,5,6,11],skiprows=1,header=None,dtype=str)
firms_df.columns= ['latitude', 'longitude','brightness','acq_date','acq_time','power']

# #if you have dataframe of firms data use this code 
new_firms_df = process_firm_data(firms_df)
# print(new_firms_df)
#merge pcd_df and firms_df together
complete_dataframe = merge_pcd_with_firms(cyclical_lag,new_firms_df)

# print(complete_dataframe)
#load model that use to predict
pcd_firms_model = pickle.load(open('xgb_model_pcd_firms.sav','rb'))
pcd_model = pickle.load(open('xgb_model_pcd.sav','rb'))


# #predict data 
predict_PMnumber1 = predict_data(complete_dataframe,'pcd_firms',station_id,pcd_firms_model)
predict_PMnumber2 = predict_data(complete_dataframe,'pcd',station_id,pcd_model)


print('3 Next Hour PM2.5 from pcd and firm model is ',predict_PMnumber1)
print('3 Next Hour PM2.5 from pcd model is ',predict_PMnumber2)