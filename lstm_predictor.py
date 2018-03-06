import trip_data_parser
import sequence_creator
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical
from keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Input
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers.merge import concatenate
from keras.models import Model
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import pandas as pd
import numpy as np
import h5py
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression

trip_data_file = "data/trip_data_jan_sept_2017_reduced.csv"
weather_data_file = "data/weather_jan_sept_2017.csv"
new_york_holidays = ['2017-01-01', '2017-01-02', '2017-01-16', '2017-01-01','2017-05-29',
                     '2017-07-04','2017-09-04','2017-10-09','2017-11-10','2017-11-11']
time_slot_size = 30 # minutes

pick_ups_by_station = trip_data_parser.get_pick_ups(trip_data_file,weather_data_file,
                                                    time_slot_size,new_york_holidays)

#pick_ups_by_station_tmp = {}
#pick_ups_by_station_tmp[72] = pick_ups_by_station[72]
#pick_ups_by_station_tmp[79] = pick_ups_by_station[79]
#pick_ups_by_station_tmp[82] = pick_ups_by_station[82]

total_stations = len(pick_ups_by_station)

########################################### LSTM - functional API ############################################

# Lock back and ahead parameters
lag = 15 # look back steps
ahead = 8 # look ahead steps
output_columns = ["pick_ups"] # target variable

# Sequential features (# pick-ups, time, day, month, weather, etc...)
sequential_data = []
# Non-sequential features (station_id, holiday)
non_sequential_data = []

print "\nPREDICTOR: Creating sequence data for each station...\n"
count = 0
# Frame data as a sequence
for station_id, pick_ups in pick_ups_by_station.iteritems():
    print "PREDICTOR: Creating sequence data for station " + str(count+1) + "/" + str(total_stations)    

    # Sequential features
    sequence_arr = sequence_creator.convert_to_sequence(pick_ups.drop(columns=["holiday"]),output_columns,lag,ahead,True)
    sequential_data.append(sequence_arr)

    # Non-sequential features
    # Holiday
    holiday_arr = pick_ups.holiday[ahead:-lag]
    holiday_arr = holiday_arr.values
    # Station id (as dummies)
    station_id_arr = np.zeros(total_stations)
    station_id_arr[count] = 1
    station_id_arr = np.tile(station_id_arr,(sequence_arr.shape[0],1)) # Repeat station vector for each sequence
    # Merge holiday vector and station matrix
    non_sequence_arr = np.hstack((station_id_arr,holiday_arr.reshape(-1,1)))
    non_sequential_data.append(non_sequence_arr)
    count += 1

# Merge all of the station sequences into a single numpy matrix
print "\nPREDICTOR: Merging sequential data into a single matrix...\n"
sequential_data = np.vstack(sequential_data)

# Merge all of the station sequences into a single numpy matrix
print "PREDICTOR: Merging non-sequential data into a single matrix...\n"
non_sequential_data = np.vstack(non_sequential_data)

# Normalize sequential data 
# (non-sequential data doesn't need to be normalized since all of the values are either 0 or 1)
print "PREDICTOR: Normalizing data...\n"
min_max_scaler = MinMaxScaler()
sequential_data = min_max_scaler.fit_transform(sequential_data)
        
# Split sequential data to input and outpu (X and y)
print "PREDICTOR: Splitting data into input and output...\n"
x_data = sequential_data[:,0:-len(output_columns)*ahead] 
y_data = sequential_data[:,-len(output_columns)*ahead:] 

# Split into training and test dataset and reshape if necessary
print "PREDICTOR: Randomly splitting data into train and test...\n"
x_train, x_test, y_train, y_test, non_sequential_train, non_sequential_test = train_test_split(x_data, y_data, non_sequential_data, test_size=0.2, random_state = 1, shuffle=True)
# Reshaping
print "PREDICTOR: Reshaping data into [samples, time steps, features]...\n"
x_train = x_train.reshape((x_train.shape[0], lag+1, x_train.shape[1]/(lag+1)))
x_test_org = x_test # Keep copy for rescaling...
x_test = x_test.reshape((x_test.shape[0], lag+1, x_test.shape[1]/(lag+1)))

# Building Keras model
print "PREDICTOR: Building Keras model...\n"

sequential_input_layer = Input(shape=(x_train.shape[1], x_train.shape[2]))

lstm_1_layer = LSTM(128)(sequential_input_layer)
repeat_layer =  RepeatVector(ahead)(lstm_1_layer)
lstm_2_layer = LSTM(128, return_sequences=True)(repeat_layer)
time_dense_layer = TimeDistributed(Dense(1))(lstm_2_layer)
activation_layer = Activation('linear')(time_dense_layer)
flatten_layer = Flatten()(activation_layer)

non_sequential_input_layer = Input(shape=(non_sequential_train.shape[1],))

# Merging the second LSTM layer and non-sequential input layer
merged = concatenate([flatten_layer, non_sequential_input_layer])
dense_1_layer = Dense(128)(merged)
dense_2_layer = Dense(128)(dense_1_layer)
output_layer = Dense(y_train.shape[1])(dense_2_layer)

# Create keras model
multi_input_model = Model(inputs=[sequential_input_layer, non_sequential_input_layer], outputs=output_layer)

# Print the model summary
print(multi_input_model.summary())

# Compile the model
multi_input_model.compile(optimizer='adam', loss='mse')

# Train the model
multi_input_model.fit([x_train, non_sequential_train], y_train, epochs=70, batch_size=1024, validation_data=([x_test, non_sequential_test], y_test), verbose=1, shuffle=True)

multi_input_model.save("model_backup.h5")

yhat = multi_input_model.predict([x_test, non_sequential_test])

y_true = min_max_scaler.inverse_transform(np.concatenate((x_test_org,y_test),axis=1))[:,-ahead:]
y_pred = min_max_scaler.inverse_transform(np.concatenate((x_test_org,yhat),axis=1))[:,-ahead:]

mse = mean_absolute_error(y_true, y_pred, multioutput = "raw_values")
print 'Test MAE', mse

########################################### TREES ############################################

# Lock back and ahead parameters
lag = 15 # look back steps
ahead = 8 # look ahead steps
output_columns = ["pick_ups"] # target variable

count = 0

samples_per_station = pick_ups_by_station.itervalues().next().shape[0] - lag - ahead
columns_per_sample = total_stations + 1 + (pick_ups_by_station.itervalues().next().shape[1] - 1)*(lag+1)+len(output_columns)*ahead

data = np.zeros((total_stations * samples_per_station,columns_per_sample))

#mins = np.array([])
#maxs = np.array([])

# Frame data as a sequence
for station_id, pick_ups in pick_ups_by_station.iteritems():
    print "PREDICTOR: Creating sequence data for station " + str(count+1) + "/" + str(total_stations)    

    sequence_arr = sequence_creator.convert_to_sequence(pick_ups.drop(columns=["holiday"]),output_columns,lag,ahead,True)
    
    data[count*total_stations:count*total_stations+samples_per_station,count] = 1
    data[count*total_stations:count*total_stations+samples_per_station,total_stations] = pick_ups.holiday[ahead:-lag].values
    data[count*total_stations:count*total_stations+samples_per_station,total_stations+1:] = sequence_arr
    
#    # Finding min and max for normalization
#    if mins.size == 0:
#        mins = sequence_arr.min(axis=0)
#    else:
#        mins = np.vstack([mins,sequence_arr.min(axis=0)]).min(axis=0)
#    
#    if maxs.size == 0:
#        maxs = sequence_arr.max(axis=0)
#    else:
#        maxs = np.vstack([maxs,sequence_arr.max(axis=0)]).max(axis=0)
    
    count += 1

# Normalize sequential data 
# (non-sequential data doesn't need to be normalized since all of the values are either 0 or 1)
print "PREDICTOR: Normalizing data...\n"
min_max_scaler = MinMaxScaler(copy=True)
data[:,total_stations+1:] = min_max_scaler.fit_transform(data[:,total_stations+1:])

# Split sequential data to input and outpu (X and y)
print "PREDICTOR: Splitting data into input and output...\n"
x_data = data[:,0:-len(output_columns)*ahead] 
y_data = data[:,-len(output_columns)*ahead:] 

# Split into training and test dataset and reshape if necessary
print "PREDICTOR: Randomly splitting data into train and test...\n"
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state = 1, shuffle=True)

# Decision tree regression ----------------------------------------
print "PREDICTOR: RFR regression...\n"

rfr_regr = RandomForestRegressor()
rfr_regr.fit(x_train, y_train)

yhat = rfr_regr.predict(x_test)
y_true = min_max_scaler.inverse_transform(np.concatenate((x_test[:,total_stations+1:],y_test),axis=1))[:,-ahead:]
y_pred = min_max_scaler.inverse_transform(np.concatenate((x_test[:,total_stations+1:],yhat),axis=1))[:,-ahead:]

mse = mean_absolute_error(y_true, y_pred, multioutput = "raw_values")
print 'DTR MAE\n', mse

# MLP regression ----------------------------------------
print "PREDICTOR: MLP regression...\n"

mlp_regr = MLPRegressor(hidden_layer_sizes=(256, 256, 256), batch_size=1024)

mlp_regr.fit(x_train,y_train)

yhat = mlp_regr.predict(x_test)
y_true = min_max_scaler.inverse_transform(np.concatenate((x_test[:,total_stations+1:],y_test),axis=1))[:,-ahead:]
y_pred = min_max_scaler.inverse_transform(np.concatenate((x_test[:,total_stations+1:],yhat),axis=1))[:,-ahead:]

mse = mean_absolute_error(y_true, y_pred, multioutput = "raw_values")
print 'MLP MAE\n', mse

# LR regression ----------------------------------------
print "PREDICTOR: LR regression...\n"

lr_regr = LinearRegression()

lr_regr.fit(x_train,y_train)

yhat = lr_regr.predict(x_test)
y_true = min_max_scaler.inverse_transform(np.concatenate((x_test[:,total_stations+1:],y_test),axis=1))[:,-ahead:]
y_pred = min_max_scaler.inverse_transform(np.concatenate((x_test[:,total_stations+1:],yhat),axis=1))[:,-ahead:]

mse = mean_absolute_error(y_true, y_pred, multioutput = "raw_values")
print 'MLP MAE\n', mse

# SVR regression ----------------------------------------
print "PREDICTOR: SVR regression...\n"

svr_regr = SVR(kernel='rbf')
svr_multi_regr = MultiOutputRegressor(svr_regr)

svr_multi_regr.fit(x_train,y_train)

yhat = svr_regr.predict(x_test)
y_true = min_max_scaler.inverse_transform(np.concatenate((x_test[:,total_stations+1:],y_test),axis=1))[:,-ahead:]
y_pred = min_max_scaler.inverse_transform(np.concatenate((x_test[:,total_stations+1:],yhat),axis=1))[:,-ahead:]

mse = mean_absolute_error(y_true, y_pred, multioutput = "raw_values")
print 'SVR MAE\n', mse

##################################################### LSTM #################################################### 

## TODO: We first perform experiments only on one test station
#station_df = pick_ups_by_station[72].copy()
#
## Frame data as a sequence
## x_data is input, y_data is output
#lag = 15
#ahead = 8
#output_columns = ["pick_ups"]
#data = sequence_creator.convert_to_sequence(station_df,output_columns,lag,ahead,True)
#
## weather and holiday data, we will use it later
##holiday_data = holiday_data[lag+1:]
##holiday_data = to_categorical(holiday_data) # From keras
##weather_data = weather_data[lag+1:]
##label_encoder = LabelEncoder() # From sklearn package
##weather_data = label_encoder.fit_transform(weather_data)
##weather_data = to_categorical(weather_data) # From keras
#
## Normalize all data
#min_max_scaler = MinMaxScaler()
#data = min_max_scaler.fit_transform(data)
#
## Split X i y data
#x_data = data[:,0:-len(output_columns)*ahead] 
#y_data = data[:,-len(output_columns)*ahead:] 
#
## Split into training and test dataset and reshape if necessary
#x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state = 1, shuffle=True)
## Reshaping
#x_train = x_train.reshape((x_train.shape[0], lag+1, x_train.shape[1]/(lag+1)))
#x_test_org = x_test
#x_test = x_test.reshape((x_test.shape[0], lag+1, x_test.shape[1]/(lag+1)))
#y_train = y_train.reshape((y_train.shape[0], ahead, 1))
#y_test_org = y_test
#y_test = y_test.reshape((y_test.shape[0], ahead, 1))
#
## WITH repeat vectors
## Create keras model for LSTM
#model = Sequential()
#model.add(LSTM(64, input_shape=(x_train.shape[1], x_train.shape[2])))
#model.add(RepeatVector(ahead))
#model.add(LSTM(64, stateful=False, return_sequences=True))
#model.add(TimeDistributed(Dense(1)))
#model.add(Activation('linear')) 
#model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
#
#model.summary()
#
#history = model.fit(x_train, y_train, epochs=100, batch_size=128, validation_data=(x_test, y_test), verbose=1, shuffle=True)#, callbacks=[es])
#
#yhat = model.predict(x_test)
#yhat = yhat.reshape((yhat.shape[0], ahead))
#
#y_true = min_max_scaler.inverse_transform(np.concatenate((x_test_org,y_test_org),axis=1))[:,-ahead:]
#y_pred = min_max_scaler.inverse_transform(np.concatenate((x_test_org,yhat),axis=1))[:,-ahead:]
#
#mae = mean_absolute_error(y_true, y_pred, multioutput = "raw_values")
#print 'Test MAE:\n', mae
#
#pyplot.plot(y_true[5,:], label='true')
#pyplot.plot(y_pred[5,:], label="predicted")


##################################################### MLP #################################################### 
#
#nn = MLPRegressor(hidden_layer_sizes=(64,128,), activation='relu', solver='adam', max_iter = 500)
#
#x_train_1 = x_train.reshape((x_train.shape[0], -1))
#y_train_1 = y_train[:,0]
#n = nn.fit(x_train_1, y_train_1)
#
#yhat = nn.predict(x_test_org)
#y_pred = min_max_scaler.inverse_transform(np.concatenate((x_test_org,yhat.reshape(yhat.shape[0],1)),axis=1))[:,-1]
#
#pyplot.plot(y_pred[0:100], label="predicted")
#pyplot.plot(y_true[0:100], label='true')
#
#
#mse = mean_squared_error(y_true, y_pred)
#print('Test MSE: %.3f' % mse)
#
################################################ RANDOM FOREST ############################################### 
#
#
#rf = RandomForestRegressor(n_estimators=200)
#rf.fit(x_train_1, y_train_1)
#yhat = rf.predict(x_test_org)
#y_pred = min_max_scaler.inverse_transform(np.concatenate((x_test_org,yhat.reshape(yhat.shape[0],1)),axis=1))[:,-1]
#
#pyplot.plot(y_pred[0:100], label="predicted")
#pyplot.plot(y_true[0:100], label='true')
#
#
#mse = mean_squared_error(y_true, y_pred)
#print('Test MSE: %.3f' % mse)
#
##################################################### SVR #####################################################
#
#svr = SVR(kernel='rbf')
#svr.fit(x_train_1, y_train_1)
#yhat = svr.predict(x_test_org)
#
#y_pred = min_max_scaler.inverse_transform(np.concatenate((x_test_org,yhat.reshape(yhat.shape[0],1)),axis=1))[:,-1]
#
#pyplot.plot(y_pred[0:100], label="predicted")
#pyplot.plot(y_true[0:100], label='true')
#
#
#mse = mean_squared_error(y_true, y_pred)
#print('Test MSE: %.3f' % mse)