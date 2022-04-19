#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import lightgbm
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import mean_squared_error
import glob
import functools
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from openpyxl import load_workbook

def Forecast(ndata, method):

    if(method == 'AR'):
        # fit model
        model = AutoReg(ndata, lags=1,old_names=False)
        model_fit = model.fit()
        # make prediction
        yhat = model_fit.predict(len(ndata), len(ndata))
    elif(method == 'MA'):
        # fit model
        model = ARIMA(ndata, order=(0, 0, 1))
        model_fit = model.fit()
        # make prediction
        yhat = model_fit.predict(len(ndata), len(ndata))
    elif(method == 'ARMA'):
        # fit model
        model = ARIMA(ndata, order=(2, 0, 1))
        model_fit = model.fit()
        # make prediction
        yhat = model_fit.predict(len(ndata), len(ndata))
    elif(method == 'ARIMA'):
        # fit model
        model = ARIMA(ndata, order=(1, 1, 1))
        model_fit = model.fit()
        # make prediction
        yhat = model_fit.predict(len(ndata), len(ndata), typ='levels')
    elif(method == 'SARIMA'):
        # fit model
        model = SARIMAX(ndata, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
        model_fit = model.fit(disp=False)
        # make prediction
        yhat = model_fit.predict(len(ndata), len(ndata))
    elif(method == 'SARIMAX'):
        # fit model
        model = SARIMAX(ndata, exog=exogdata, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
        model_fit = model.fit(disp=False)
        # make prediction
        yhat = model_fit.predict(len(ndata), len(ndata), exog=[6])
    elif(method == 'VARMAX'):
         print('varmax')
    elif(method == 'SES'):
        # fit model
        model = SimpleExpSmoothing(ndata)
        model_fit = model.fit()
        # make prediction
        yhat = model_fit.predict(len(ndata), len(ndata))
    elif(method == 'HWES'):
        # fit model
        model = ExponentialSmoothing(ndata)
        model_fit = model.fit()
        # make prediction
        yhat = model_fit.predict(len(ndata), len(ndata))
    
    return yhat

def LightGBM():
    
    #lightgbm regression
    row_count2 = x_train.shape[0]
    split_point2 = int(row_count2*test_ratio)
    X, X_test = x_train[:split_point2], x_train[split_point2:]
    Y, Y_test = y_train[:split_point2], y_train[split_point2:]
    train_data = lightgbm.Dataset(X, label=Y)
    test_data = lightgbm.Dataset(X_test, label=Y_test)
    parameters = {'objective': 'regression',
              'boosting': 'gbdt',
              'metric': 'rmse',
              'learning_rate': 0.1,
              'seed' : 42}
    model = lightgbm.train(parameters,
                           train_data,
                           valid_sets=test_data,
                           num_boost_round=5000,
                           early_stopping_rounds=100)
    return model

def LR():
    reg = LinearRegression()
    reg.fit(x_train, y_train)
    return reg

def date(d,cv_type):
    if(cv_type =='last 5%'):
        rcount = d.shape[0]
        splitpoint = int(rcount*test_ratio)
        xtest = d[splitpoint:]
        return xtest.Date
        
    else:#-----
        rcount = d.shape[0]
        splitpoint = int(rcount*test_ratio)
        xtest = d[splitpoint:]
        return xtest.Date
    
def make_sliding(df, N):
    dfs = [df.shift(-i).applymap(lambda x: [x]) for i in range(0, N+1)]
    return functools.reduce(lambda x, y: x.add(y), dfs)

prior_file_dict = dict()
test_ratio = 85/100 # 1 %
#while(1):
    
file_dict = dict()
# Load data
extension = 'csv'
files = glob.glob('*.{}'.format(extension))
files = [ x for x in files if ".ai" not in x ]
#for file in files:# files without .ai name
#    if('.ai' in file):
#        files.remove(file)

# files is empty
if not files:
    print('No csv File seen.')
    #break
    sys.exit()

targets = [' ZigZag Value']
regression_names = ['LinearRegression','Lightgbm']
forecast_names = ['AR','MA','ARMA','ARIMA','SARIMA','SARIMAX','SES','HWES']
#forecast_names = ['AR']
#forecast_names = ['MA']
#result_file = pd.DataFrame(columns = ['File_Name','LiteGBM', 'LR','AR','MA','ARMA','ARIMA','SARIMA','SES','HWES','Rank'])
result_file = pd.DataFrame(columns = ['File_Name','AR','MA','ARMA','ARIMA','SARIMA','SARIMAX','SES','HWES','LinearRegression','Lightgbm','best'])
#result_file = pd.DataFrame(columns = ['File_Name','AR','best'])

for j in range(len(files)):
    data = pd.read_csv(files[j], parse_dates=['Date'])
    tempdata = pd.read_csv(files[j])
    
    #test_date
    test_date = date(tempdata,'last 5%')
    # new features
    data['month'] = data.Date.dt.month
    #data['week'] = data.Date.dt.week
    data['week_day'] = data.Date.dt.weekday
    data['year'] = data.Date.dt.year
    data['days_in_month'] = data.Date.dt.days_in_month
    data = data.drop("Date", axis=1)

    #results = pd.Series()
    #results = np.append(results, files[j])
    results = pd.DataFrame(columns = ['File_Name','AR','MA','ARMA','ARIMA','SARIMA','SARIMAX','SES','HWES','LinearRegression','Lightgbm','best'])
    #results = pd.DataFrame(columns = ['File_Name','AR','best'])
    results = results.append(pd.Series(), ignore_index=True)
    results.File_Name = files[j]
    
    file_dict[files[j]] = data.shape[0]
    if files[j] not in prior_file_dict:

        text = ""
        for target in targets:
            x_data = data.loc[:,data.columns != target]
            y_data = data[target]

            # test 15 % last
            row_count = data.shape[0]
            split_point = int(row_count*test_ratio)
            x_train, x_test = x_data[:split_point], x_data[split_point:]
            y_train, y_test = y_data[:split_point], y_data[split_point:]

            #-----------------
            temp_error = x_test[' ZigZag Length'].mean()/100000
            for forecast_name in forecast_names:
                new_data = y_train
                exogdata = x_train['week_day']
                
                y_pred = pd.Series()
                for m in range(row_count - split_point):# predict next test data
                    #ai_predicted.loc[ai_predicted.shape[0]-1,'Date'] = tempdata.Date[prior_file_dict[files[j]]+count] # add Date


                    # predict next value
                    if(target == ' ZigZag Value'):
                        #mdl = LinearRegression()
                        #mdl.fit(x_train, y_train)
                        y_hat = Forecast(new_data,forecast_name)
                    else:
                        train_data = lightgbm.Dataset(X, label=Y)
                        test_data = lightgbm.Dataset(X_test, label=Y_test)
                        parameters = {'objective': 'regression','boosting': 'gbdt','metric': 'rmse','learning_rate': 0.1,'seed' : 42}
                        mdl = lightgbm.train(parameters,train_data,valid_sets=test_data,num_boost_round=5000,early_stopping_rounds=100)
                        #mdl = regressions[1]
                    #y_pred = y_pred.append(mdl.predict(x_test2))
                    #ttemp = mdl.predict(x_test2)
                    y_pred = np.append(y_pred, y_hat)
                    #x_test3.c14 = y_pred2[0]
                    #y_pred3 = mdl.predict(x_test3)
                    #if(target == 'ZigZag'):
                    #    next_zigzag2 = y_pred2
                        #next_zigzag3 = y_pred3

                    new_data = np.append(new_data, y_test[y_test.index[0]+m])
                    exogdata = np.append(exogdata, x_test['week_day'][x_test.index[0]+m])
                    
                #temp_y_test= y_test.to_numpy()
                #percent_error = (abs((y_pred[1:])-(temp_y_test[:-1])) / abs((temp_y_test[1:])-(temp_y_test[:-1])) ) *100
                #percent_error = [ x for x in percent_error if np.isfinite(x)]
                #percent_error = np.mean(percent_error)
                #results[forecast_name]= percent_error # mean
                
                #percent_error= ((abs(y_test - y_pred) / abs(y_test))*100)
                #percent_error = np.mean(percent_error)
                #results[forecast_name]= percent_error # mean
                #results = np.append(results, np.mean(percent_error))
                rmse = np.sqrt(mean_squared_error(y_test,y_pred))
                percent_error = (rmse/temp_error)*100
                results[forecast_name]= percent_error # mean
                if(target == ' ZigZag Value'):
                    zigzag = y_pred
                elif(target == 'ZigZag Length'):
                    zigzaglength = y_pred
                #percent_error= ((abs(y_test - y_pred) / abs(y_test))*100) 
                #percent_error = percent_error.mean()
                print("RMSE = " , rmse)
                text = text + "\nRoot mean squared error for "+ forecast_name +' for '+target +" = " + str(rmse) +"\n"
                text = text +'---------------------------------------------------------'
                text = text + "\nPercent error for "+  forecast_name +' for ' + target +" = " + str(percent_error) +"\n"
                text = text +'---------------------------------------------------------'
            
            #-----------------
            for ii in range(2):
                new_data = data[:split_point] 
                window_size = 14
                x = make_sliding(new_data,window_size)
                x = x.loc[:,new_data.columns == target]
                x[['c1','c2','c3','c4','c5',
                   'c6','c7','c8','c9','c10',
                   'c11','c12','c13','c14','c15']] = pd.DataFrame(x[target].tolist(), index= x.index)
                x = x.drop(target, axis=1)
                #x_test3 = x[x.shape[0]-window_size+1:x.shape[0]-window_size+2]
                x = x[:x.shape[0]-window_size+1]

                y_pred = pd.Series()
                for m in range(row_count - split_point):# predict next test data
                    #ai_predicted.loc[ai_predicted.shape[0]-1,'Date'] = tempdata.Date[prior_file_dict[files[j]]+count] # add Date


                    # predict next value

                    x_test2 = x[x.shape[0]-1:].drop("c15", axis=1)
                    #x_test3 = x_test3.drop("c15", axis=1)
                    train2 = x[:x.shape[0]-1]
                    y_train = train2.c15
                    x_train = train2.drop("c15", axis=1)

                    # validation of lightgbm
                    row_count2 = x_train.shape[0]
                    split_point2 = int(row_count2*test_ratio)
                    X, X_test = x_train[:split_point2], x_train[split_point2:]
                    Y, Y_test = y_train[:split_point2], y_train[split_point2:]

                    if(ii == 0):
                        mdl = LinearRegression()
                        mdl.fit(x_train, y_train)
                    else: 
                        train_data = lightgbm.Dataset(X, label=Y)
                        test_data = lightgbm.Dataset(X_test, label=Y_test)
                        parameters = {'objective': 'regression','boosting': 'gbdt','metric': 'rmse','learning_rate': 0.1,'seed' : 42}
                        mdl = lightgbm.train(parameters,train_data,valid_sets=test_data,num_boost_round=5000,early_stopping_rounds=100)
                        #mdl = regressions[1]
                    #y_pred = y_pred.append(mdl.predict(x_test2))
                    ttemp = mdl.predict(x_test2)
                    y_pred = np.append(y_pred, ttemp[0])
                    #x_test3.c14 = y_pred2[0]
                    #y_pred3 = mdl.predict(x_test3)
                    #if(target == 'ZigZag'):
                    #    next_zigzag2 = y_pred2
                        #next_zigzag3 = y_pred3

                    x.loc[x.shape[0]-1,'c15'] = y_test[y_test.index[0]+m]
                    tempp2 = pd.DataFrame()
                    tempp= x[x.shape[0]-1:]
                    tempp = tempp.drop("c1", axis=1)
                    tempp2[['c1','c2','c3','c4','c5',
                       'c6','c7','c8','c9','c10',
                       'c11','c12','c13','c14']] = tempp
                    tempp2['c15'] = np.nan
                    #x= np.append(x,tempp2)
                    x = pd.concat([x,tempp2] , ignore_index=True)
                #-------------------


                rmse = np.sqrt(mean_squared_error(y_test,y_pred))
                #-------
                percent_error = (rmse/temp_error)*100
                #results[forecast_name]= percent_error # mean
                
                if(target == 'ZigZag'):
                    zigzag = y_pred
                elif(target == 'ZigZag Length'):
                    zigzaglength = y_pred
                
                #temp_y_test= y_test.to_numpy()
                #percent_error = (abs((y_pred[1:])-(temp_y_test[:-1])) / abs((temp_y_test[1:])-(temp_y_test[:-1])) ) *100
                #percent_error = [ x for x in percent_error if np.isfinite(x) ]
                #percent_error = np.mean(percent_error)
                
                #percent_error= ((abs(y_test - y_pred) / abs(y_test))*100) 
                #percent_error = percent_error.mean()
                
                results[regression_names[ii]]= percent_error
                print("RMSE = " , rmse)
                text = text + "\nRoot mean squared error for "+ regression_names[ii] +' for '+ target +" = " + str(rmse) +"\n"
                text = text +'---------------------------------------------------------'
                text = text + "\nPercent error for "+ regression_names[ii] +' for '+ target +" = " + str(percent_error) +"\n"
                text = text +'---------------------------------------------------------'
                       
            
            '''
            # predict next value
            next_new_data = y_data

            if(target == ' ZigZag Value'):# Linear regression
                #mdl = LinearRegression()
                #mdl.fit(x_train, y_train)
                y_pred2 = Forecast(next_new_data,forecast_name)
                next_new_data = np.append(next_new_data, y_pred2)
                y_pred3 = Forecast(next_new_data,forecast_name)
            else: # Lightgbm
                train_data = lightgbm.Dataset(X, label=Y)
                test_data = lightgbm.Dataset(X_test, label=Y_test)
                parameters = {'objective': 'regression','boosting': 'gbdt','metric': 'rmse','learning_rate': 0.1,'seed' : 42}
                mdl = lightgbm.train(parameters,train_data,valid_sets=test_data,num_boost_round=5000,early_stopping_rounds=100)
                #mdl = regressions[1]


            if(target == ' ZigZag Value'):
                next_zigzag2 = y_pred2
                next_zigzag3 = y_pred3
            text = text + "\n two Next prediction value for "+ target +" = "+str(y_pred2)+" and "+ str(y_pred3)+"\n"
            text = text +'---------------------------------------------------------'
            '''
        file = open(files[j][:-4] +'_Log.txt', 'w')
        file.write(text)
        file.close()
    
    results['best'] = min(results.values[0][1:-1]) #------------MIN----------
    result_file = result_file.append(results, ignore_index=True)

# Create Result file
#result_file = pd.DataFrame(columns = ['File_Name','LiteGBM', 'LR','AR','MA','ARMA','ARIMA','SARIMA','SES','HWES','Rank'])

#result_file.File_Name = test_date
#ai_file = ai_file.append(pd.Series(), ignore_index=True)
#ai_file = ai_file.append(pd.Series(), ignore_index=True)
#ai_file.ZigZag = zigzag
#zigzaglength = zigzaglength.astype(int)
#ai_file['ZigZag Length'] = zigzaglength

result_file = result_file.sort_values(by=['best'], ascending=True)#saoodi
result_file = result_file.drop(columns=['best'])
result_file.to_csv('result.ai.csv', index=False)
read_file = pd.read_csv ('result.ai.csv')
read_file.to_excel ('result.ai.xlsx', index = None, header=True)


#print("Waiting Time")
#prior_file_dict = file_dict
#time.sleep(30)

