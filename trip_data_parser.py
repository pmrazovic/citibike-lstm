import pandas as pd
import numpy as np

def get_pick_ups(trip_data_file,weather_data_file,time_slot_size,holidays):
	# Reading trip data and filter pick ups only
	pick_ups_df = pd.read_csv(trip_data_file)
	# Set datetime index and sort by its value
	pick_ups_df.index = pd.to_datetime(pick_ups_df["starttime"])
	pick_ups_df.sort_index
	# Remove all columns except 'start station id'
	pick_ups_df.drop(pick_ups_df.columns.difference(['start station id']), 1, inplace=True)
	# Add new column with the total number of pick ups
	pick_ups_df["pick_ups"] = 1
	# Rename columns
	pick_ups_df.columns = ["station_id","pick_ups"]

	# Reading weather data
	weather_data = pd.read_csv(weather_data_file, index_col = 0)
	# Set datetime index and sort by its value
	weather_data.index = pd.to_datetime(weather_data.index)

	# We will create a separate dataframe for each station
	pick_ups_by_station = {}
	# New index with 30 min time slices
	new_index = pd.date_range(start = '2017-01-01 00:00:00', end = "2017-09-30 23:59:59", freq=str(time_slot_size)+'Min')

	# Process pick ups for each station
	for station_id, station_df in pick_ups_df.groupby('station_id'):
	    # Create a new dataframe for the current station
	    new_df = pd.DataFrame()
	    # Copy pick ups column and index
	    new_df["pick_ups"] = station_df.pick_ups.values
	    new_df.index = station_df.index
	    new_df.sort_index
	    # Group and sum pick ups by 30 min time slices
	    new_df = new_df.groupby(pd.Grouper(freq=str(time_slot_size)+'Min', base=0, label='left')).sum()
	    # Replace NaN with zeros
	    new_df['pick_ups'].fillna(0, inplace=True)
	    # Update index, so we can have rows with 0 pick ups for missing time slices
	    new_df = new_df.reindex(index = new_index, fill_value = 0)
	    
	    # Add column for day of week
	    new_df["day_of_week"] = new_df.index.dayofweek
	    # Add column for month
	    new_df["month"] = new_df.index.month
	    # Add column for timeslot, i.e., minutes passed since midnight
	    new_df["minutes_from_midnight"] = new_df.index.hour*60 + new_df.index.minute
	    # Add column for holiday
	    new_df["holiday"] = 0
	    new_df.loc[np.isin(new_df.index.strftime('%Y-%m-%d'), holidays), 'holiday'] = 1
	    
	    # Adding columns for weather data
	    # Find weather data for the corresponding time slots
	    idx = np.searchsorted(weather_data.index, new_df.index) - 1
	    new_df["condition"] = weather_data.iloc[idx].condition.values
	    new_df["temperature"] = weather_data.iloc[idx].temperature.values
	    new_df["wind_speed"] = weather_data.iloc[idx].wind_speed.values
	    new_df["humidity"] = weather_data.iloc[idx].humidity.values
	    new_df["visability"] = weather_data.iloc[idx].visability.values
	    
	    # Add station dataframe to dictionary
	    pick_ups_by_station[station_id] = new_df

	return pick_ups_by_station