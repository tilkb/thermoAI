from __future__ import absolute_import
import sys
sys.path.append(sys.path[0] +"/..")

from simulator import energyCost
from simulator import weather
from simulator import building
from simulator import simulator
from controller import scheduler

def test_constant_price():
    cost=energyCost.ConstantEnergyCost(10.1)
    c1=cost.get_cost_at(10)
    c2=cost.get_cost_at(2000)
    assert(c1==c2)
    assert(c1==10.1)

def test_night_daytime_cost():
    timeslots=[(0,300,4.5),
               (301, 1200,8.8),
               (1201,1440,4.5)]
    cost=energyCost.DailyEnergyCost(timeslots)
    assert(cost.get_cost_at(150)==4.5)
    assert(cost.get_cost_at(500)==8.8)
    assert(cost.get_cost_at(1310)==4.5)

def test_weather_smoke():
    w=weather.Weather()
    w.get_out_temperature(1000)
    w.get_out_temperature(10000)

def test_model():
    m = building.Building(100000,100,1000,20.0)
    before_temp = m.get_inside_temperature()
    for i in range(100):
        m.step(0.0, -10.0)
    after_temp = m.get_inside_temperature()
    assert(before_temp>after_temp)

def test_load_model():
    m = building.Building.from_json("simulator/config/conf1.json")
    before_temp = m.get_inside_temperature()
    m.step(0.0, 10.0)


def test_daily_scheduler():
    timeslots=[(0,300,19,26),
               (301, 1200,13,26),
               (1201,1440,19,26)]
    sch= scheduler.DailyScheduler(timeslots)
    assert(sch.get_target(150)==(19,26))
    assert(sch.get_target(500)==(13,26))
    assert(sch.get_target(1310)==(19,26))

def test_weekly_scheduler():
    timeslots1=[(0,300,19,26),
               (301, 1200,13,26),
               (1201,1440,19,26)]
    timeslots2=[(0,1440,20,24)]
    monday_friday= scheduler.DailyScheduler(timeslots1)
    saturday_sunday= scheduler.DailyScheduler(timeslots2)
    week =[monday_friday]*5 + [saturday_sunday]*2
    sch=scheduler.WeeklyScheduler(week)
    assert(sch.get_target(150)==(19,26))
    assert(sch.get_target(1940)==(13,26))
    assert(sch.get_target(4190)==(19,26))
    assert(sch.get_target(7500)==(20,24))
    

def test_simulator_smoke():
    m = building.Building(100000,100,1000,20.0,)
    w=weather.Weather()
    timeslots=[(0,300,4.5),
               (301, 1200,8.8),
               (1201,1440,4.5)]
    cost=energyCost.DailyEnergyCost(timeslots)
    timeslots2=[(0,480,20,24),
               (480, 1000,12,26),
               (1001,1440,20,24)]
    sch= scheduler.DailyScheduler(timeslots2)
    sim = simulator.Simulator(m, sch, cost, w, time_step_size_minute=5)
    sim.reset()
    sim.step(200)
    sim.step(300)
    sim.step(400)
    sim.reset()


def test_simulator_load():
    sim = simulator.Simulator.from_json("simulator/config/simulation1.json")
    sim.step(100)
    sim.reset()
