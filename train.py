import pickle
from simulator.simulator import Simulator
from controller.PID.PIDController import PIDController
from controller.RL.DDPG import DDPGController

def train_pid(sim_file):
    sim = Simulator.from_json(sim_file)
    pid = PIDController()
    pid.train(sim, response_step_count=10)
    with open('controller/saved/PID.pkl', 'wb') as f:
        pickle.dump(pid, f)
    print("PID saved...")

def train_rl(sim_file):
    sim = Simulator.from_json(sim_file)
    ppo = PPOController(sim)
    ppo.train(sim)
    ppo.save('controller/saved/PPO/')
    print("PPO saved...")
    ddpg = DDPGController(sim)
    ddpg.train(sim)
    ddpg.save('controller/saved/DDPG2/')
    print("DDPG saved...")


if __name__== "__main__":
    train_pid("simulator/config/simulation1.json")
    train_rl("simulator/config/simulation1.json")
