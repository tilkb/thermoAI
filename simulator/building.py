import json
from datetime import timedelta

class Building():
    """A simple Building Energy Model.

    Consisting of one thermal capacity and one resistance, this model is derived from the
    hourly dynamic model of the ISO 13790. It models heating energy demand only.

    Parameters:
        * heat_mass_capacity:           capacity of the building's heat mass [J/K]
        * heat_transmission:            heat transmission to the outside [W/K]
        * maximum_heating_power:        [W] (>= 0)
        * initial_building_temperature: building temperature at start time [℃]
        * time_step_size:               [timedelta]
    """

    @classmethod
    def from_json(cls, conf_file, time_step_size=timedelta(minutes=5)):
        with open(conf_file) as f:
            data = json.load(f)
        return cls(data["heat_mass_capacity"], data["heat_transmission"],
                    data["maximum_heating_power"], data["initial_building_temperature"], time_step_size)

    def __init__(self, heat_mass_capacity, heat_transmission,
                 maximum_heating_power,
                 initial_building_temperature, time_step_size=timedelta(minutes=5)):
        if maximum_heating_power < 0:
            raise ValueError("Maximum heating power [W] must not be negative.")
        self.__heat_mass_capacity = heat_mass_capacity
        self.__heat_transmission = heat_transmission
        self.__maximum_heating_power = maximum_heating_power
        self.current_temperature = initial_building_temperature
        self.__time_step_size = time_step_size

    def get_max_heating_power(self):
        return self.__maximum_heating_power

    def step(self,outside_temperature, heating_power):
        """Performs building simulation for the next time step.
        Parameters:
            * outside_temperature: [℃]
            * heating_power: heating power [W]
        Return:
            * used power [W]
            * temperature in the next timestep [℃]
        """
        if heating_power>self.__maximum_heating_power:
            heating_power=self.__maximum_heating_power
        
        dt_by_cm = self.__time_step_size.total_seconds() / self.__heat_mass_capacity
        next_temperature = (self.current_temperature * (1 - dt_by_cm * self.__heat_transmission) +
                dt_by_cm * (heating_power + self.__heat_transmission * outside_temperature))
        self.current_temperature = next_temperature
        return heating_power, next_temperature

    def get_inside_temperature(self):
        """
        Return the curent temperure in [℃]
        """
        return self.current_temperature

    def set_inside_temperature(self, temperature):
        """
        Set the curent temperure in [℃]
        """
        self.current_temperature = temperature