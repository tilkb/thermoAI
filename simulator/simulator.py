import json
from datetime import timedelta
import numpy as np
from simulator.weather import Weather
from simulator.energyCost import DailyEnergyCost
from simulator.building import Building
from controller.scheduler import DailyScheduler

class Simulator:
    """Simulator class simulate the heating period. 
        Every module can be replaced to more elaberated one if it is required
        Parameters:
            * model: the heat model of the building
            * schedule: controls the target temperature interval
            * price_scheduler: provide the price of the energy in every timestep
            * time_step: the time resolution of the simulation [minute]
            * prev_step_feature: number of feature from the past
            * next_step_feature: number of feature in the future
            * temperature_noise_sigma: noise during the temperature measurement
            * power_noise_sigma: noise during the controlling
            * weather_prediction_sigma: noise of the weather prediction
    """

    @classmethod
    def from_json(cls, conf_file):
        with open(conf_file) as (f):
            conf = json.load(f)
        weather = Weather(conf['weather_file'])
        price_scheduler = DailyEnergyCost(conf['energy_cost'])
        scheduler = DailyScheduler(conf['schedule'])
        model = Building.from_json(conf['model_file'], timedelta(minutes=conf['simulation_step_size_minute']))
        return cls(model, scheduler, price_scheduler, weather, conf['simulation_step_size_minute'], conf['prev_states_feature'], conf['prev_states_feature'], conf['temperature_noise_sigma'], conf['power_noise_sigma'], conf['weather_prediction_sigma'])

    def __init__(self, model, schedule, price_scheduler, weather, time_step_size_minute=5, prev_states_feature=20, next_states_feature=20, temperature_noise_sigma=0.1, power_noise_sigma=4, weather_prediction_sigma=0.8):
        self.time_step_size_minute = time_step_size_minute
        self.scheduler = schedule
        self.price_scheduler = price_scheduler
        self.weather = weather
        self.heat_model = model
        self.prev_states_count = prev_states_feature
        self.next_states_count = next_states_feature
        self.temperature_noise_sigma = temperature_noise_sigma
        self.power_noise_sigma = power_noise_sigma
        self.weather_prediction_sigma = weather_prediction_sigma
        self._init_temperature = self.heat_model.get_inside_temperature()
        self.reset()

    def reset(self):
        self.historical_consuption = []
        self.historical_inside_temp = []
        self.historical_outside_temp = []
        self.current_time = 0
        self.total_cost = 0
        self._last_heating_power = 0.0
        self.heat_model.set_inside_temperature(self._init_temperature)
        return self._get_state()

    def step(self, action):
        """
        action: is the heating power in the next time step
        """
        power = max(0, np.random.normal(loc=action, scale=self.power_noise_sigma))
        heating_power, curr_inside_temp = self.heat_model.step(self.weather.get_out_temperature(self.time_step_size_minute * self.current_time), action)
        self._last_heating_power = heating_power
        temp_from, temp_to = self.scheduler.get_target(self.time_step_size_minute * self.current_time)
        not_satisfy_penalty = 0 if temp_from <= curr_inside_temp and temp_to >= curr_inside_temp else -1000000
        reward = -power * self.price_scheduler.get_cost_at(self.time_step_size_minute * self.current_time) + not_satisfy_penalty
        self.total_cost += power * self.price_scheduler.get_cost_at(self.time_step_size_minute * self.current_time)
        self.current_time += 1
        self.historical_consuption.append(heating_power)
        inside_temp = np.random.normal(loc=self.heat_model.get_inside_temperature(), scale=self.temperature_noise_sigma)
        self.historical_inside_temp.append(inside_temp)
        outside_temp = np.random.normal(loc=self.weather.get_out_temperature(self.time_step_size_minute * self.current_time), scale=self.temperature_noise_sigma)
        self.historical_outside_temp.append(outside_temp)
        done = self.weather.get_timeseries_length_minutes() <= self.time_step_size_minute * self.current_time
        return done, reward, self._get_state()

    def _get_state(self):
        future_required_temperatures = [ self.scheduler.get_target(self.time_step_size_minute * time) for time in range(self.current_time, self.current_time + self.next_states_count) ]
        future_outside_temperatures = [ self.weather.get_out_temperature(self.time_step_size_minute * time) for time in range(self.current_time, self.current_time + self.next_states_count) ]
        future_outside_temperatures = [ np.random.normal(loc=temp, scale=self.weather_prediction_sigma * ((1 + idx) / len(future_outside_temperatures))) for idx, temp in enumerate(future_outside_temperatures) ]
        future_energy_cost = [ self.price_scheduler.get_cost_at(self.time_step_size_minute * time) for time in range(self.current_time, self.current_time + self.next_states_count) ]
        previous_outside_temperatures = self.historical_outside_temp[-self.prev_states_count:]
        previous_inside_temperatures = self.historical_inside_temp[-self.prev_states_count:]
        previous_energy_consuption = self.historical_consuption[-self.prev_states_count:]
        return future_required_temperatures, future_outside_temperatures, future_energy_cost, previous_outside_temperatures, previous_inside_temperatures, previous_energy_consuption

