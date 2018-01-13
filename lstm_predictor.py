import pandas as pd
import keras
import trip_data_parser
import sequence_creator
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

trip_data_file = "data/trip_data_jan_sept_2017_reduced.csv"
weather_data_file = "data/weather_jan_sept_2017.csv"
new_york_holidays = ['2017-01-01', '2017-01-02', '2017-01-16', '2017-01-01','2017-05-29',
                     '2017-07-04','2017-09-04','2017-10-09','2017-11-10','2017-11-11']
time_slot_size = 30 # minutes

pick_ups_by_station = trip_data_parser.get_pick_ups(trip_data_file,weather_data_file,
                                                    time_slot_size,new_york_holidays)

# We first perform experiments only on one test station
station_df = pick_ups_by_station[72].copy()

# For now, we just remove non-teporal variable "holiday"
station_df.drop(columns=["holiday"],inplace=True)

# For now, we encode categorical variable "condition" with integer encoding
# Later we will use one-hot encoding
label_encoder = LabelEncoder() # From sklearn package
station_df.condition = label_encoder.fit_transform(station_df.condition.values)

# Normalize all data
min_max_scaler = MinMaxScaler()
station_df = pd.DataFrame(min_max_scaler.fit_transform(station_df), columns=station_df.columns)

# Frame data as a sequence
# x_data is input, y_data is output
lag = 15
x_data, y_data = sequence_creator.convert_to_sequence(station_df,["pick_ups"],lag,1)

# Convert data into np array and split into traning and test
x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(x_data, y_data, test_size=0.10, shuffle=False)

# Reshape input to be 3D [samples, timesteps, features]
x_data_train = x_data_train.values.reshape((x_data_train.shape[0], lag+1, x_data_train.shape[1]/(lag+1)))
x_data_test = x_data_test.values.reshape((x_data_test.shape[0], lag+1, x_data_test.shape[1]/(lag+1)))
