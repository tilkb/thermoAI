import datetime

class EnergyCost:
    """
        Simple energy cost query model
        The resolution of the time is one minute
    """
    def get_cost_at(self,time):
        """
        Return the cost at the given time
        """
        pass

class ConstantEnergyCost(EnergyCost):
    def __init__(self,price):
        self.price = price

    def get_cost_at(self,time):
        return self.price

class DailyEnergyCost(EnergyCost):
    def __init__(self,prices):
        """
        prices: [(time_from, time_to, price)]
        """
        self.prices = prices
        prices.sort(key=lambda x:x[0])

    def get_cost_at(self, time):
        day_time=time%(24*60)
        for start, end, cost in self.prices:
            if start<=day_time and end>=day_time:
                return cost


