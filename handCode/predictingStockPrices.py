#Predicting stock prices with python(LSTM) Tensorflo 1.6
from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

data_source = 'kaggle'

if data_source == 'alphavantage':
	api_key = '[API key]'
	ticker = "AAL"
	url_string = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s"%(ticker,api_key)
	file_to_save = 'stock_market_data-2018%s.csv'%ticker
	if not os.path.exists(file_to_save):
		with urllib.request.urlopen(url_string) as url:
			data = json.loads(url.read().decode())
			
			#extract stock market data
			data = data['Time Series (Daily)']
			df = pd.DataFrame(columns=['Date', 'Low', 'High', 'Close', 'Open'])

			for k, v in data.items():
				date = dt.datetime.strptime(k, '%Y-%m-%d')
				data_row = [date.date(), float(v['3. low']), float(v['2. high']), float(v['4. close']), float(v['1. open'])]
				df.loc[-1,:] = data_row
				df.index = df.index + 1
		print('Data saved to : %s'%file_to_save)
		df.to_csv(file_to_save)
	
	#if the data is alreade there, simply load it from the csv file
	else:
	    print('File already exists. Loading data from CSV')
	    df = pd.read_csv(file_to_save)
else:

    # ====================== Loading Data from Kaggle ==================================
    # You will be using HP's data. Feel free to experiment with other data.
    # But while doing so, be careful to have a large enough dataset and also pay attention to the data normalization
	df = pd.read_csv(os.path.join('Stocks','hpq.us.txt'),delimiter=',',usecols=['Date','Open','High','Low','Close'])

	print('Loaded data from the kaggle repository')	



#Data exploration

df = df.sort_values('Date')
df.head()

#Visualize data
plt.figure(figsize = (18,9))
plt.plot(range(df.shape[0]), (df['low']+df['high'])/2.0)
plt.xticks(range(0,df.shape[0],500), df['Date'].loc[::500],rotation=45)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Mid Price', fontsize=18)
plt.show()

#Split data into a training set and test set
#Calculate the mid prices from the highest and lowest
high_prices = df.loc[:,'High'].as_matrix()
low_prices = df.loc[:,'Low'].as_matrix()
mid_prices = (high_prices+low_prices)/2.0

#actually split the test and train sets
train_data = mid_prices[:11000]
test_data = mid_prices[11000:]

#Normalize data
scaler = MinMaxScaler()
train_data = train_data.reshape(-1,1)
test_data = test_data.reshape(-1,1)

