import pandas as pd
import numpy as np
import sys

def get_pick_ups(trip_data_file,weather_data_file,time_slot_size,holidays):
    # Reading trip data and filter pick ups only
    print("PARSER: Reading trip data from CSV...\n")
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

    print("PARSER: Counting pick-ups at each station...\n")
    pick_ups_grouped = pick_ups_df.groupby('station_id')
    total_stations = len(pick_ups_grouped)
    station_counter = 1
    # Process pick ups for each station
    for station_id, station_df in pick_ups_grouped:
        print ("PARSER: Couting pick-ups for station " + str(station_counter) + "/" + str(total_stations))
        sys.stdout.flush()
        station_counter += 1
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
        new_df["temperature"] = weather_data.iloc[idx].temperature.values
        new_df["wind_speed"] = weather_data.iloc[idx].wind_speed.values
        new_df["humidity"] = weather_data.iloc[idx].humidity.values
        new_df["visability"] = weather_data.iloc[idx].visability.values
        new_df["condition"] = weather_data.iloc[idx].condition.values
        
        # Binning weather condition and introducing dummies
        levels = pd.DataFrame(["rainy","cloudy","foggy"]) # Possible values for condition
        cloudy_conds = ["Clear","Partly Cloudy", "Scattered Clouds", "Mostly Cloudy", "Haze", "Overcast"]
        foggy_conds = ["Fog","Mist"]
        new_df.loc[-new_df.condition.isin(cloudy_conds + foggy_conds),'condition'] = "rainy"
        new_df.loc[new_df.condition.isin(cloudy_conds),'condition'] = "cloudy"
        new_df.loc[new_df.condition.isin(foggy_conds),'condition'] = "foggy"
        
        condition_dummies = pd.get_dummies(new_df.condition.append(levels), prefix="condition")
        condition_dummies.drop(condition_dummies.tail(len(levels)).index,inplace=True) # Drop added levels     
        new_df = new_df.join(condition_dummies)
        
        # Transform time variables to introduce cyclic features
        new_df["sin_time"] = np.sin(((2*np.pi)/1440.0)*new_df.minutes_from_midnight)
        new_df["cos_time"] = np.cos(((2*np.pi)/1440.0)*new_df.minutes_from_midnight)
        new_df["sin_dow"] = np.sin(((2*np.pi)/7.0)*new_df.day_of_week)
        new_df["cos_dow"] = np.cos(((2*np.pi)/7.0)*new_df.day_of_week)
        # Drop original time features
        new_df.drop(columns=["minutes_from_midnight","day_of_week","condition"],inplace=True)
        
        # Add station dataframe to dictionary
        pick_ups_by_station[station_id] = new_df

    return pick_ups_by_station

def get_drop_offs(trip_data_file,weather_data_file,time_slot_size,holidays):
    # Reading trip data and filter drop offs only
    print("PARSER: Reading trip data from CSV...\n")
    drop_offs_df = pd.read_csv(trip_data_file)
    # Set datetime index and sort by its value
    drop_offs_df.index = pd.to_datetime(drop_offs_df["stoptime"])
    drop_offs_df.sort_index
    # Remove all columns except 'end station id'
    drop_offs_df.drop(drop_offs_df.columns.difference(['end station id']), 1, inplace=True)
    # Add new column with the total number of drop_offs
    drop_offs_df["drop_offs"] = 1
    # Rename columns
    drop_offs_df.columns = ["station_id","drop_offs"]
    
    # Reading weather data
    weather_data = pd.read_csv(weather_data_file, index_col = 0)
    # Set datetime index and sort by its value
    weather_data.index = pd.to_datetime(weather_data.index)
    
    # We will create a separate dataframe for each station
    drop_offs_by_station = {}
    # New index with 30 min time slices
    new_index = pd.date_range(start = '2017-01-01 00:00:00', end = "2017-09-30 23:59:59", freq=str(time_slot_size)+'Min')

    print("PARSER: Counting drop-offs at each station...\n")
    drop_offs_grouped = drop_offs_df.groupby('station_id')
    total_stations = len(drop_offs_grouped)
    station_counter = 1
    # Process drop offs for each station
    for station_id, station_df in drop_offs_grouped:
        print ("PARSER: Couting drop-offs for station " + str(station_counter) + "/" + str(total_stations))
        sys.stdout.flush()
        station_counter += 1
        # Create a new dataframe for the current station
        new_df = pd.DataFrame()
        # Copy drop offs column and index
        new_df["drop_offs"] = station_df.drop_offs.values
        new_df.index = station_df.index
        new_df.sort_index
        # Group and sum drop_offs by 30 min time slices
        new_df = new_df.groupby(pd.Grouper(freq=str(time_slot_size)+'Min', base=0, label='left')).sum()
        # Replace NaN with zeros
        new_df['drop_offs'].fillna(0, inplace=True)
        # Update index, so we can have rows with 0 drop_offs for missing time slices
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
        new_df["temperature"] = weather_data.iloc[idx].temperature.values
        new_df["wind_speed"] = weather_data.iloc[idx].wind_speed.values
        new_df["humidity"] = weather_data.iloc[idx].humidity.values
        new_df["visability"] = weather_data.iloc[idx].visability.values
        new_df["condition"] = weather_data.iloc[idx].condition.values
        
        # Binning weather condition and introducing dummies
        levels = pd.DataFrame(["rainy","cloudy","foggy"]) # Possible values for condition
        cloudy_conds = ["Clear","Partly Cloudy", "Scattered Clouds", "Mostly Cloudy", "Haze", "Overcast"]
        foggy_conds = ["Fog","Mist"]
        new_df.loc[-new_df.condition.isin(cloudy_conds + foggy_conds),'condition'] = "rainy"
        new_df.loc[new_df.condition.isin(cloudy_conds),'condition'] = "cloudy"
        new_df.loc[new_df.condition.isin(foggy_conds),'condition'] = "foggy"
        
        condition_dummies = pd.get_dummies(new_df.condition.append(levels), prefix="condition")
        condition_dummies.drop(condition_dummies.tail(len(levels)).index,inplace=True) # Drop added levels     
        new_df = new_df.join(condition_dummies)
        
        # Transform time variables to introduce cyclic features
        new_df["sin_time"] = np.sin(((2*np.pi)/1440.0)*new_df.minutes_from_midnight)
        new_df["cos_time"] = np.cos(((2*np.pi)/1440.0)*new_df.minutes_from_midnight)
        new_df["sin_dow"] = np.sin(((2*np.pi)/7.0)*new_df.day_of_week)
        new_df["cos_dow"] = np.cos(((2*np.pi)/7.0)*new_df.day_of_week)
        # Drop original time features
        new_df.drop(columns=["minutes_from_midnight","day_of_week","condition"],inplace=True)
        
        # Add station dataframe to dictionary
        drop_offs_by_station[station_id] = new_df

    return drop_offs_by_station