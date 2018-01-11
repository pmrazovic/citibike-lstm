import pandas as pd
import keras
import trip_data_parser

trip_data_file = "data/trip_data_jan_sept_2017_reduced.csv"
weather_data_file = "data/weather_jan_sept_2017.csv"
new_york_holidays = ['2017-01-01', '2017-01-02', '2017-01-16', '2017-01-01','2017-05-29',
                     '2017-07-04','2017-09-04','2017-10-09','2017-11-10','2017-11-11']
time_slot_size = 30 # minutes

pick_ups_by_station = trip_data_parser.get_pick_ups(trip_data_file,weather_data_file,
                                                    time_slot_size,new_york_holidays)

test_station = pick_ups_by_station[72]

