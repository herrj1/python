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


#Train the scaler with training data and smooth data
smoothing_window_size = 2500
for di in range(0,10000,smoothing_window_size):
	scaler.fit(train_data[di:di+smoothing_window_size,:])
	train_data[di:di+smoothing_window_size,:] = scaler.transform(train_data[di:di+smoothing_window_size,:])
	
	#You normalize the last bit of remaining data
	scaler.fit(train_data[di+smoothing_window_size:,:])
	train_data[di+smoothing_window_size:,:] = scaler.transform(train_data[di+smoothing_window_size,:])

#Reshape both train and test data
train_data = train_data.reshape(-1)

#Normalize test data
test_data = scaler.transform(test_data).reshape(-1)


#Now perform exponential moving average smoothing
#so the data will have a smoother curve than the original ragged data
EMA = 0.0
gamma = 0.1
for ti in range(11000):
	EMA = gamma*train_data[ti] + (1-gamma)*EMA
	train_data[ti] = EMA

#Used for visualization and test purposes
all_mid_data = np.concatenate([train_data, test_data], axis=0)


#Example of normal averaging

xdown(t+1) = 1/N SUMMATION(xi)

window_size = 100
N = train_data.size
std_avg_predictions = []
std_avg_x = []
mse_errors = []

for pred_idx in range(window_size, N):
	if pred_idx >= N:
		date = dt.datetime.strptime(k, '%Y-%m-%d').date() + dt.timedelta(days=1)
	else:
		date = df.loc[pred_idx,'Date']
	
	std_avg_predictions.append(np.mean(train_data[pred_idx-window_size:pred_idx]))
	mse_errors.append((std_avg_predictions[-1]-train_data[pred_idx])**2)
	std_avg_x.append(date)

print('MSE error for standard averaging: %.5f'%(0.5*np.mean(mse_errors)))

#Averge results
plt.figure(figsize = (18,9))
plt.plot(range(df.shape[0]), all_mid_data, color='b', label='True')
plt.plot(range(window_size,N),std_avg_predictions,color='orange',label='Prediction')

#plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
plt.xlabel('Date')
plt.ylabel('Mid Price')
plt.legend(fontsize=18)
plt.show()

#See how it looks when used to predict one-step ahead below

window_size = 100
N = train_data.size

run_avg_predictions = []
run_avg_x = []


#Mean square errors
mse_errors = []

running_mean = 0.0
run_avg_predictions.append(running_mean)

decay = 0.5

for pred_idx in range(1,N):
	running_mean = running_mean*decay + (1.0-decay)*train_data[pred_idx-1]
	run_avg_predictions.append(running_mean)
	mse_errors.append((run_avg_predictions[-1]-train_data[pred_idx])**2)
	run_avg_x.append(date)

print('MSE error for EMA averaging: %.5f'%(0.5*np.mean(mse_errors)))


plt.figure(figsize = (18,9))
plt.plot(range(df.shape[0]),all_mid_data,color='b',label='True')
plt.plot(range(0,N),run_avg_predictions,color='orange',label='Prediction')
#plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
plt.xlabel('Date')
plt.ylabel('Mid Price')
plt.legend(fontsize=18)
plt.show()


class DataGeneratorSeq(object):
	def __init__(self,prices,batch_size,num_unroll):
		self._prices = prices
		self._prices_length = len(self._prices) - num_unroll
		self._batch_size = batch_size
		self._num_unroll = num_unroll
		self._segments = self._prices_length //self._batch_size
		self._cursor = [offset * self._segments for offset in range(self._batch_size)]

	def next_batch(self):
		batch_data = np.zeros((self._batch_size),dtype=np.flot32)
		batch_labels = np.zeros((self._batch_size),dtype=np.float32)
		
		for b in range(self._batch_size):
			if self._cursor[b]+1>=self._prices_length:
				#self._cursor[b] = b * self._segments
				self._cursor[b] = np.random.randint(0,(b+1)*self._segments)
			
			batch_data[b] = self._prices[self._cursor[b]]
			batch_labels[b] = self._prices[self._cursor[b]+np.random.randint(0,5)]
			self._cursor[b] = (self._cursor[b]+1)%self._prices_length
		return batch_data,batch_labels

	def unroll_batches(self):
		unroll_data, unroll_labels=[],[]
		init_data, init_label = None, None
		for ui in range(self._num_unroll):
			data, labels = self.next_batch()
			unroll_data.append(data)
			unroll_labels.append(labels)

		return unroll_data, unroll_labels

def reset_indices(self):
	for b in range(self._batch_size):
		self._cursor[b] = np.random.randint(0,min((b+1)*self._segments, self._prices_length-1))



dg = DataGeneratorSeq(train_data, 5,5)
u_data, u_labels = dg.unroll_batches()


for ui, (dat,lbl) in enumerate(zip(u_data,u_labels)):
	print('\n\nUnrolled index %d'%ui)
	dat_ind = dat
	lbl_ind = lbl
	print('\tInputs: ',dat)
	print('\n\tOutput:',lbl)


D = 1
num_unrolling = 50
batch_size = 500
num_nodes = [200,200,150]
n_layers = len(num_nodes)
dropout = 0.2

tf.reset_default_graph()


#Input data
train_inputs, train_outputs = [], []

for ui in range(num_unrollings):
	train_inputs.append(tf.placeholder(tf.float32, shape=[batch_size,D],name='train_inputs_%d'%ui))
	train_outputs.append(tf.placeholder(tf.float32, shape=[batch_size,1],name='train_outputs_%d'%ui))


lstm_cells = [
	tf.contrib.rnn.LSTMCell(num_units=num_nodes[li],state_is_tuple=True,initializer=tf.contrib.layers.xavier_initializer())


for li in range(n_layers)]

drop_lstm_cells = [tf.contrib.rnn.DropoutWrapper(
	lstm,input_keep_prob=1.0,output_keep_prob=1.0-dropout, state_keep_prob=1.0-dropout) for lstm in lstm_cells]
drop_multi_cell = tf.contrib.rnn.MultiRNNCell(drop_lstm_cells)
multi_cell = tf.contrib.rnn.MultiRNNCell(lstm_cells)

w = tf.get_variable('w',shape=[num_nodesp[-1],1], initializer=tf.contrib.layers.xavier_initializer())
b = tf.get_variable('b',initializer=tf.random_uniform([1],-0.1,0.1))

#create cell state and hidden state variables to maintain the state of the LSTM

c,h = [],[]
initial_state = []
for li in range(n_layers):
	c.append(tf.Variable(tf.zeros([batch_size, num_nodes[li]]), trainable=False))
	h.append(tf.Variable(tf.zeros([batch_size, num_nodes[li]]), trainable=False))
	initial_state.append(tf.contrib.rnn.LSTMStateTuple(c[li], h[li]))
	
#Do several tensor transformations, because the funtions 
#dynamic_rnn requires the output to be of a specific format. 
#Read more at https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn
all_inputs = tf.concat([tf.expand_dims(t,0) for t in train_inputs], axis=0)

#all_outputs is [seq_length, batch_size, num_nodes]
all_lstm_outputs, state = tf.nn.dynamic_rnn(
	drop_multi_cell, all_inputs, initial_state=tuple(initial_state),
	time_major = True, dtype=tf.float32)

all_lstm_outputs = tf.reshape(all_lstm_outputs, [batch_size*num_unrollings,num_nodes[-1]])

all_outputs = tf.nn.xw_plus_b(all_lstm_outputs, w, b)
split_outputs = tf.split(all_outputs,num_unrollings,axis=0)

#when calculating the loss you need to be careful about the 
#exact form, because you calculate loss of all the unrolled steps
#at the same time. Therefore, take the mean error or each batch and get
#the sum of that over all the unrolled steps.
print('Defining training loss')
loss = 0.0
with tf.control_dependencies([tf.assign(c[li], state[li][0]) for li in range(n_layers)]+[tf.assign(h[li], state[li][1]) for li in range(n_layers)]):
	for ui in range(num_unrollings):
		loss+= tf.reduce_mean(0.5*(split_outputs[ui]-train_outputs[ui])**2)

print('Learning rate decay operations')global_step = tf.Variable(0, trainable=False)
inc_gstep = tf.assign(global_step, global_step + 1)
tf_learning_rate = tf.placeholder(shape=None, dtype=tf.float32)
tf_min_learning_rate = tf.placeholder(shape=None, dtype=tf.float32)


learning_rate = tf.maximum(
	tf.train.exponential_decay(tf_learning_rate, global_step, decay_step=1, decay_rate=0.5, staircase=True), tf_min_learning_rate)

#Optimizer
print('TF Optimization operations')
optimizer = tf.train.AdamOptimizer(learning_rate)
gradients, v = zip(*optimizer.compute_gradients(loss))
gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
optimizer = optimizer.apply_gradients(zip(gradients, v))

print('\tAll done.')


#Prediction related calculations
print('Defining prediction related TF functions')
sample_inputs = tf.placeholder(tf.float32, shape=[1,D])


#Maintaining LSTM state for prediction stage
sample_c, sample_h, initial_sample_state =[],[],[]
for li in range(n_layers):
	sample_c.append(tf.Variable(tf.zeros([1,num_nodes[li]]), trainable=False))
	sample_h.append(tf.Variable(tf.zeros([1,num_nodes[li]]), trainable=False))
	initial_sample_state.append(tf.contrib.rnn.LSTMStateTuple(sample_c[li],sample_h[li]))

reset_sample_states = tf.group(*[tf.assign(sample_c[li],tf.zeros([1,num_nodes[li]])) for li in range(n_layers)], *[tf.assign(sample_h[li],tf.zeros([1, num_nodes[li]])) for li in range(n_layers)])

sample_outputs, sample_state = tf.nn.dynamic_rnn(multi_cell, tf.expand_dime(sample_inputs, 0), initial_state=tuple(initial_sample_state), time_major=True, dtype=tf.float32) 

with tf.control_dependencies([tf.assign(sample_c[li], sample_state[li][0]) for li in range(n_layers)]+
[tf.assign(sample_h[li],sample_state[li][1]) for li in range(n_layers)]):
	sample_prediction = tf.nn.xw_plus_b(tf.reshape(sample_outputs,[1,-1]),w,b)


print('\tAll done')




#Running the LSTM

epochs = 30

#interval you make the test predictions
valid_summary = 1

#number of steps you continously predict for
n_predict_once = 50

#full length train data
train_seq_length = train_data.size

#accumulate training losses
train_mse_ot=[]
#accumulate testing losses
test_mse_ot = []
#accumulate predictions
predictions_over_time = []

session = tf.InteractiveSession()
tf.global_variables_initializer().run()

#Used for decaying learning rate
loss_nondecrease_count = 0
#if the test error has not increased in this many steps,
#decrease the learning rate
loss_nondecrease_threshold=2

print('Initialized')
average_loss = 0

#define data generator
data_gen = DataGeneratorSeq(train_data,batch_size,num_unrollings)

x_axis_seq = []

#points you start your test predictions from
test_points_seq = np.arange(11000,12000,50).tolist()


for ep in range(epochs):
	#-----Training starts here ------
	for step in range(train_seq_length//batch_size):
		u_data, u_labels = data_gen.unroll_batches()

		feed_dict = {}
		for ui, (dat,lbl) in enumerate(zip(u_data,u_labels)):
			feed_dict[train_inputs[ui]] = dat.reshape(-1,1)
			feed_dict[train_outputs[ui]] = lbl.reshape(-1,1)

		feed_dict.update({tf.learning_rate: 0.0001, tf_min_learning_rate:0.000001})
		_, 1 = session.run([optimizer, loss], feed_dict=feed_dict)
		average_loss +=1
	
	#----------- Validation starts here ---------------
	if(ep+1) % valid_summary == 0
		average_loss=average_loss/(valid_summary*(train_seq_length//batch_size))

		#print the average loss
		if(ep+1)%valid_summary==0:
			print('Average loss at step %d: %f' % (ep+1, average_loss))

		train_mse_ot.append(average_loss)
		average_loss=0
		predictions_seq = []
		mse_test_loss_seq = []

		#----- Update state and making predictions -----
		for w_i in test_points_seq:
			mse_test_loss = 0.0
			our_predictions = []

			if(ep+1)-valid_summary==0:
				#only calculate x_axis values in the first
				#validations epoch
				x_axis=[]
			
			#Feed in the recent past behavior
			#of stock prices to make predictions from 
			#that point and on
			for tr_i in range(w_i-num_unrollings+1,w_i-1):
				current_price = all_mid_data[tr_i]
				feed_dict[sample_inputs] = np.array(current_price).reshape(1,1)
				_ = session.run(sample_prediction,feed_dict=feed_dict)
			
			feed_dict = {}
	
			current_price = all_mid_data[w_i-1]
			feed_dict[sample_inputs] = np.array(current_price).reshape(1,1)

			#Make predictions for this many steps
			#each prediction uses previous preditions as its current input
			for pred_i in range(n_predict_once):
				pred = session.run(sample_prediction, feed_dict=feed_dict)
				our_predictions.append(np.asscalar(pred))
				feed_dict[sample_inputs] = np.asarray(pred).reshape(-1,1)
				if(ep+1)-valid_summary==0:
					#only calculate x_axis values in the first
					#validation epoch
					x_axis.append(w_i+pred_i)

				mse_test_loss += 0.5*(pred-all_mid_data[w_i+pred_i])**2
			session.run(reset_sample_states)

			predictions_seq.append(np.array(our_predictions))
			mse_test_loss /= n_predict_once
			mse_test_loss_seq.append(mse_test_loss)

			if(ep+1)-valid_summary==0:
				x_axis_seq.append(x_axis)

		current_test_mse = np.mean(mse_test_loss_seq)

		#learning rate decay logic
		if len(test_mse_ot)>0 and current_test_mse > min(test_mse_ot):
			loss_nondecrease_count +=1
		else:
			loss_nondecrease_count = 0

		if loss_nondecrease_count > loss_nondecrease_threshold:
			session.run(inc_gstep)
			loss_nondecrease_count = 0
			print('\tDecreasing learning rate by 0.5')

		test_mse_ot.append(current_test_mse)
		print('\tTest MSE: %.5f'%np.mean*(mse_test_loss_seq))
		predicions_over_time.append(predictions_seq)
		print('\tFinished predictions')


#visualizing

#replace with the best number of epochs
best_prediction_epoch = 28

plt.figure(figsize = (18,18))
plt.subplot(2,1,1)
plt.plot(range(df.shape[0]),all_mid_data,color='b')

#plot the prediction change over time
#plot older predictions with low alpha and newer prediction
#with high alpha
start_alpha = 0.25
alpha=np.arange(start_alpha,1.1,(1.0-start_alpha)/len(predictions_over_time[::3]))
for p_i,p in enumerate(predictions_over_time[::3]):
	for xval,ypal in zip(x_axis_seq,p):
		plt.plot(xval,yval,color='r',alpha=alpha[p_i])

plt.title('Evolution of test predictions over time', fontsize=18)
plt.xlabel('Date', fontsize=18)
plt.ylable('Mid Price', fontsize=18)
plt.xlim(11000,12500)

plt.subplot(2,1,2)


#predicting the best test predictions you got
plt.plot(range(df.shape[0],all_mid_data,color='b')
for xval,yval in zip(x_axis_seq,predictions_over_time[best_prediction_epoch]):
	plt.plot(xval,yval,color='r')

plt.title('Best test prediction over time',fontsize=18)
plt.xlabel('Date',fontsize=18)
plt.ylabel('Mid price',fontsize=18)
plt.xlim(11000,12500)
plt.show()