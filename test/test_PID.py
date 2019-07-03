from __future__ import absolute_import
import sys
sys.path.append(sys.path[0] +"/..")

from controller.PID.PIDController import PID
from controller.PID.PIDController import PIDController
from simulator import simulator

def test_pid_getter():
    pid = PID(1,2,3)
    p,i,d = pid.components
    assert(p==1)
    assert(i==2)
    assert(d==3)

def test_pid_setter():
    pid = PID(0,0,0)
    pid.tuning(1,2,3)
    p,i,d = pid.components
    assert(p==1)
    assert(i==2)
    assert(d==3)

def test_pid_control():
    pid = PID(100,0,0)
    power = pid.update(10, 15)
    assert(power>0.0)
    power2 = pid.update(21,20)
    assert(power2<=0.0)
    

def test_pidcontroller_control_smoke():
    pid = PIDController()
    sim = simulator.Simulator.from_json("simulator/config/simulation1.json")
    for i in range(30):
        _, _, (future_required_temperatures, future_outside_temperatures, future_energy_cost, previous_outside_temperatures, previous_inside_temperatures, previous_energy_consuption) =sim.step(0)
        power = pid.control(future_required_temperatures, future_outside_temperatures, future_energy_cost, previous_outside_temperatures, previous_inside_temperatures, previous_energy_consuption)

def test_pidcontroller_parameter_training_smoke():
    pid = PIDController()
    sim = simulator.Simulator.from_json("simulator/config/simulation1.json")
    pid.train(sim, 20)

def test_full_control():
    pid = PIDController()
    sim = simulator.Simulator.from_json("simulator/config/simulation1.json")
    pid.train(sim, 20)
    prev_error=1000
    power=0
    sim.reset()
    for i in range(4):
        for j in range(4):
            _, _, (future_required_temperatures, future_outside_temperatures, future_energy_cost, previous_outside_temperatures, previous_inside_temperatures, previous_energy_consuption) =sim.step(power)
            err =abs(previous_inside_temperatures[-1]-future_required_temperatures[0][0])
            power = pid.control(future_required_temperatures, future_outside_temperatures, future_energy_cost, previous_outside_temperatures, previous_inside_temperatures, previous_energy_consuption)
        assert(err<=prev_error+0.08)
        prev_error=err

    

