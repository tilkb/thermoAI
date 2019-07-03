import pandas as pd

class Weather:
    """
       Weather class provides temperature data for a given time
       time resultion is one minute
       Parameter:
            *weather_file: the csv file which, contains hourly temperature data
    """
    def __init__(self, weather_file="simulator/weather_data/temp_2016_2017_basel.csv"):
        self.temperatures = pd.read_csv(weather_file,sep=';')
    def get_out_temperature(self, time):
        idx=time//60
        if idx<self.temperatures.shape[0]:
            return self.temperatures['Temperature'].values[idx]
        else:   
            return 0.0

    def get_timeseries_length_minutes(self):
        return 60*len(self.temperatures)