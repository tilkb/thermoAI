class Scheduler:
    def get_target(self, time):
        pass


class DailyScheduler(Scheduler):
    def __init__(self, values):
        """
        Values: list of (start,end, min temperature, max temperature)
        """
        self.values=values
        self.values.sort(key=lambda x:x[0])

    def get_target(self, time):
        """
        time:int--> number of minutes since the begining of simulation
        """
        day_time=time%(24*60)
        for start, end, min_temp, max_temp in self.values:
            if start<=day_time and end>=day_time:
                return min_temp, max_temp


class WeeklyScheduler(Scheduler):
    def __init__(self, daily_schedulers):
        """
        daily_values: a DalyScheduler for each day
        """
        self.daily_schedulers = daily_schedulers

    def get_target(self, time):
        week_minute = time % (7*24*60)
        day = week_minute // (60*24)
        return self.daily_schedulers[day].get_target(time)