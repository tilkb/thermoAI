import tensorflow as tf
from controller.RL.utils.common import ReplayMemory

class DDPGController:
    def __init__(self):
        self.ddpg=DDPG()

    def control(self,future_required_temperatures, future_outside_temperatures, future_energy_cost, previous_outside_temperatures, previous_inside_temperatures, previous_energy_consuption):
        state = future_required_temperatures + future_outside_temperatures + future_energy_cost + previous_outside_temperatures + previous_inside_temperatures + previous_energy_consuption
        tf_state = tf.constant(state, name="State")
        return self.ddpg.control(tf_state)

    def train(self, simulator):
        self.ddpg.train(simulator)

class DDPG:
    def __init__(self):
        self.actor = tf.keras.Sequential([
                    tf.keras.layers.Dense(40,activation=tf.nn.relu),
                    tf.keras.layers.Dense(1,activation=tf.nn.relu)
                    ], name="Actor")
        self.critic= tf.keras.Sequential([
                    tf.keras.layers.Dense(40,activation=tf.nn.relu),
                    tf.keras.layers.Dense(1)
                    ], name="Critic")

    
    def train(self, simulator, episode=1000,):
        for ep in range(episode):
            simulator.reset()
            done = False
            while not(done):
                print(simulator.get_concated_features())
                state = tf.constant(simulator.get_concated_features())
                action = self.actor(state)
                done, reward, _ = simulator.step(action)
            act = self.actor()
            

    def control(self, state):
        return self.actor(state)




    

        

