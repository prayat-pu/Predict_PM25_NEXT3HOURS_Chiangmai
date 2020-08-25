# Predict_PM25_NEXT3HOURS_Chiangmai
Feature selection and problem analysis
https://medium.com/@bosssds65/pm2-5-analyzed-in-chiang-mai-thailand-e5c8573847d

# Predict Part
Image below are all process of this project
![image](https://user-images.githubusercontent.com/20863536/91233430-61f97500-e75b-11ea-8dbb-1011ae3a424f.png)

1) การหาหรือเก็บข้อมูลที่เกี่ยวข้องกับ PM2.5 (Data Gathering)
2) การสำรวจลักษณะของข้อมูล (Data Visualization)
3) การทำความสะอาดข้อมูล (Cleaning Data)
4) การเติมแต่งข้อมูลและปรับแต่งข้อมูล (Data Engineering)
5) การฝึกฝนแบบจำลอง (Training Models)
6) การปรับแต่งพารามิเตอร์ (Hyper Parameters Tuning)
7) การทำนายค่า PM2.5 ในอีกสามชั่วโมงข้างหน้า (Prediction Next 3 Hour PM2.5 Values)

(Data Gathering)
  get data from (Pollution Control Department: PCD), www.Wunderground.com, firms.modaps.eosdis.nasa.gov (FIRMS)

# Abstract:
  The particle matters 2.5 (PM2.5) density in Chiang Mai is much higher than the standards set by the World Health Organization (WHO), resulting in many ill effects to Chiang Mai, including the health problems of local population and economic problems related to tourism setback. This problem has occurred for many years and is likely to continue in the future. Currently, there is no clear solution to the problem and thus the problem won’t be resolved in a short time. So, the researcher can recognize the problems and realize the effect that will occur in the future. This research, used air pollution data, meteorological data and wildfire hotspots that occur at the study area, which is Chiang Mai for analysis, aim to find the factors that affect the change of PM2.5 and use this information to predict PM2.5 in the next three hours. For prediction models, we use Multiple Linear Regression, Random Forest, Extreme Gradient Boosting and Artificial Neural Network, and will compare the efficiencies among the models. Another goal is to find the best model that can be used to predict the density of PM2.5.
From using cleansed data to train each model, the result shows the Extreme Gradient Boosting is the most effective model by using performance indicators as follows 1. Root Mean Square Error (RMSE) = 6.31769 µg/m3, Mean Absolute Error (MAE) = 4.1775 µg/m3 and Mean Absolute Percentage Error (MAPE) = 18.63 %, R2 = 0.91318. 

# Result
image below is predict pm2.5 level with 35t and 36t station at 2020-04-13
![image](https://user-images.githubusercontent.com/20863536/91234032-ac2f2600-e75c-11ea-80bb-3637fa1cdf9e.png)
![Predict PM2 5 Value with PCD FIRMS Dataset by Xgboost at 36T 2020-04-14](https://user-images.githubusercontent.com/20863536/91234096-c79a3100-e75c-11ea-8c03-3043207d12b5.png)
