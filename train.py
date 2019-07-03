import pickle
from simulator.simulator import Simulator
from controller.PID.PIDController import PIDController

def train_pid(sim_file):
    sim = Simulator.from_json(sim_file)
    pid = PIDController()
    pid.train(sim)
    with open('controller/saved/PID.pkl', 'wb') as f:
        pickle.dump(pid, f)
    print("saved PID saved...")

def train_rl():
    pass



if __name__== "__main__":
    train_pid("simulator/config/simulation1.json")
    train_rl()
