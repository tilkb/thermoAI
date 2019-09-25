import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import pickle
from controller.RL.utils.common import ReplayMemory, Transition

class DDPGController:
    def __init__(self, simulator):
        for i in range(50):
            simulator.step(0)
        self.ddpg=DDPG(len(simulator.get_concated_features()), 0.0, simulator.heat_model.get_max_heating_power())
    def control(self,future_required_temperatures, future_outside_temperatures, future_energy_cost, previous_outside_temperatures, previous_inside_temperatures, previous_energy_consuption):
        future_min = [x[0] for x in future_required_temperatures]
        future_max = [x[1] for x in future_required_temperatures]
        state = future_min + future_max + future_outside_temperatures + future_energy_cost + previous_outside_temperatures + previous_inside_temperatures + previous_energy_consuption
        tf_state = tf.constant(state, name="State")
        tf_state = tf.reshape(tf_state,(1,-1))
        return self.ddpg.control(tf_state)[0,0]

    def q_estimation(,future_required_temperatures, future_outside_temperatures, future_energy_cost, previous_outside_temperatures, previous_inside_temperatures, previous_energy_consuption,action)
        future_min = [x[0] for x in future_required_temperatures]
        future_max = [x[1] for x in future_required_temperatures]
        state = future_min + future_max + future_outside_temperatures + future_energy_cost + previous_outside_temperatures + previous_inside_temperatures + previous_energy_consuption
        tf_state = tf.constant(state, name="State")
        tf_state = tf.reshape(tf_state,(1,-1))
        tf_action = tf.constant([action], name="Action")
        return self.ddpg.q_value(tf_state, tf_action).numpy()[0,0]


    def train(self, simulator):
        gamma =0.7
        #collect data from PID controlling
        simulator.reset()
        for t in range(simulator.prev_states_count):
                simulator.step(0)
        done = False
        with open('controller/saved/PID.pkl', 'rb') as f:
            pid = pickle.load(f)
        if os.path.isfile('controller/saved/DDPG/DDPG_actor.h5') and os.path.isfile('controller/saved/DDPG/DDPG_critic.h5'): 
            self.load('controller/saved/DDPG/')
        else:
            power = 0
            state_data=[]
            act_data = []
            while not(done):
                done,reward, (future_required_temperatures, future_outside_temperatures,future_energy_cost,
                previous_outside_temperatures, previous_inside_temperatures, previous_energy_consuption) = simulator.step(power)
                power = pid.control(future_required_temperatures, future_outside_temperatures,future_energy_cost,
                previous_outside_temperatures, previous_inside_temperatures, previous_energy_consuption)
                normalized_power = max(0,min(power,simulator.heat_model.get_max_heating_power()))
                act_data.append(normalized_power)
                state = simulator.get_concated_features()
                state_data.append(state)

            state_data=np.array(state_data,dtype=np.float32)
            act_data = np.expand_dims(np.array(act_data,dtype=np.float32),axis=1)
            dataset = tf.data.Dataset.from_tensor_slices((state_data,act_data))
            self.ddpg.pretrain_actor(dataset)
            #fit q value
            power = 0
            state_data=[]
            q_data = []
            act_data = []
            done = False
            simulator.reset()
            for i in range(simulator.prev_states_count):
                simulator.step(0.0)
            while not(done):
                done,reward, (future_required_temperatures, future_outside_temperatures,future_energy_cost,
                previous_outside_temperatures, previous_inside_temperatures, previous_energy_consuption) = simulator.step(power)
                power = self.control(future_required_temperatures, future_outside_temperatures,future_energy_cost,
                previous_outside_temperatures, previous_inside_temperatures, previous_energy_consuption)

                power = max(0,min(power,simulator.heat_model.get_max_heating_power()))
                act_data.append(normalized_power)
                state = simulator.get_concated_features()
                state_data.append(state)
                q_data.append(reward)
            for i in range(len(q_data)-2,-1,-1):
                q_data[i]+=gamma*q_data[i+1]

            state_data=np.array(state_data,dtype=np.float32)
            act_data = np.expand_dims(np.array(act_data,dtype=np.float32),axis=1)
            state_action = np.concatenate((state_data,act_data), axis=1)
            q_data = np.expand_dims(np.array(q_data,dtype=np.float32),axis=1)
            q_value_dataset = tf.data.Dataset.from_tensor_slices((state_action,q_data))
            self.ddpg.pretrain_q(q_value_dataset)
        
        self.ddpg.train(simulator, simulator.prev_states_count, gamma=gamma)
    def save(self,path):
        self.ddpg.actor.save(path+'DDPG_actor.h5')
        self.ddpg.critic.save(path+'DDPG_critic.h5')
        
    def load(self,path):
        self.ddpg.actor = tf.keras.models.load_model(path+'DDPG_actor.h5')
        self.ddpg.critic = tf.keras.models.load_model(path+'DDPG_critic.h5')
class DDPG:
    def __init__(self, feature_size, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value
        self.actor = tf.keras.Sequential([
                    tf.keras.layers.Dense(300, activation=tf.nn.relu, input_shape=(feature_size,)),
                    tf.keras.layers.Dense(300, activation=tf.nn.relu),
                    tf.keras.layers.Dense(1)
                    ], name="Actor")
        self.critic= tf.keras.Sequential([
                    tf.keras.layers.Dense(300,activation=tf.nn.relu, input_shape=(feature_size+1,)),
                    tf.keras.layers.Dense(300,activation=tf.nn.relu),
                    tf.keras.layers.Dense(1)
                    ], name="Critic")

    def pretrain_q(self,q_value_dataset,epoch=100, objective='mse'):
        q_value_dataset = q_value_dataset.batch(64)
        self.critic.compile(loss=objective, optimizer='adam')
        self.critic.fit(q_value_dataset, epochs=epoch)

    def pretrain_actor(self,dataset, epoch=100, objective='mae'):
        dataset =dataset.batch(64)
        if objective == 'adversarial':
            disc = tf.keras.Sequential([
            tf.keras.layers.Dense(300, activation=tf.nn.relu),
            tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
            ], name="discriminator")
            print("Adversarial pretrain")
            optimizer = tf.keras.optimizers.Adam()
            bce = tf.keras.losses.BinaryCrossentropy()
            for ep in tqdm(range(epoch)):
                sum_loss_disc = tf.zeros((1),dtype=tf.dtypes.float32)
                sum_loss_gen = tf.zeros((1),dtype=tf.dtypes.float32)
                for state, action in dataset:
                    #Discriminator training
                    with tf.GradientTape() as tape:
                        predicted_action = self.actor(state)
                        fake_pred = disc(tf.concat([state,predicted_action],axis=1))
                        real_pred = disc(tf.concat([state,action],axis=1))
                    
                        fake_loss =bce(tf.zeros_like(fake_pred),fake_pred)
                        real_loss = bce(tf.ones_like(real_pred), real_pred)
                        loss = (fake_loss + real_loss) / 2.0
                    sum_loss_disc = sum_loss_disc + tf.math.reduce_sum(loss)
                    grad_disc = tape.gradient(loss, disc.trainable_variables)
                    optimizer.apply_gradients(zip(grad_disc, disc.trainable_variables))
                    #Generator training
                    with tf.GradientTape() as tape:
                        predicted_action = self.actor(state)
                        fake_pred = disc(tf.concat([state,predicted_action],axis=1))
                        loss = bce(tf.ones_like(fake_pred),fake_pred)
                    sum_loss_gen = sum_loss_gen + tf.math.reduce_sum(loss)
                    grad_generator = tape.gradient(loss,self.actor.trainable_variables)
                    optimizer.apply_gradients(zip(grad_generator, self.actor.trainable_variables))
                print('Discriminator loss:', sum_loss_disc)
                print('Generator loss:', sum_loss_gen)

                
        else:
            self.actor.compile(loss=objective, optimizer='adam')
            self.actor.fit(dataset, epochs=epoch)

    
    def train(self, simulator, init_step=0,episode=10, batch_size=256, gamma=0.95):
        gamma = tf.constant([gamma])
        optimizer = tf.keras.optimizers.Adam(0.000002)
        replay_memory = ReplayMemory(400)
        commulative_reward_history = []
        target_actor = tf.keras.models.clone_model(self.actor)
        target_critic = tf.keras.models.clone_model(self.critic)
        noise = 0.003
        noise_decay=0.98
        polyak = tf.constant([0.95])
        for ep in tqdm(range(episode)):
            noise = noise*noise_decay
            simulator.reset()
            #reach initial state with enough history
            for i in range(init_step):
                simulator.step(0.0)

            done = False
            sum_reward=0.0
            while not(done):
                #act in the simulator
                state = tf.constant(simulator.get_concated_features(), dtype=tf.float32)
                state = tf.reshape(state,(1,-1))
                action = (self.actor(state)).numpy() 
                action =  np.clip(np.random.normal(loc = action, scale = noise*(self.max_value-self.min_value)),self.min_value, self.max_value)
                #normalized_action = action
                #action = action * (self.max_value-self.min_value) + self.min_value
                done, reward, _ = simulator.step(action[0,0])
                sum_reward += reward
                next_state = tf.constant(simulator.get_concated_features(), dtype=tf.float32)
                if not(done):
                    replay_memory.push(state,tf.constant(action, dtype=tf.float32),next_state, tf.constant(reward, dtype=tf.float32))
            
                #training 
                if len(replay_memory)>=batch_size:
                    transitions = replay_memory.sample(batch_size)
                    batch = Transition(*zip(*transitions))
                    state = tf.reshape(tf.stack(batch.state,axis=0),(batch_size,-1))
                    action = tf.reshape(tf.stack(batch.action,axis=0),(batch_size,-1)) 
                    next_state = tf.reshape(tf.stack(batch.next_state,axis=0),(batch_size,-1))
                    reward = tf.reshape(tf.stack(batch.reward,axis=0),(batch_size,-1))
                    
                    with tf.GradientTape() as tape:
                        y_pred = self.q_value(self.critic,state, action)
                        #print('Q:',y_pred)
                        action_next = self.actor(next_state)
                        y_target = reward + gamma * self.q_value(target_critic,next_state, target_actor(next_state))
                        l = tf.keras.losses.Huber()
                        loss = l(y_pred,y_target)
                    grad_critic = tape.gradient(loss, self.critic.trainable_variables)
                    optimizer.apply_gradients(zip(grad_critic, self.critic.trainable_variables))

                    with tf.GradientTape() as tape:
                        q = - tf.math.reduce_mean(self.q_value(self.critic, state, self.actor(state)))
                    grad_actor = tape.gradient(q, self.actor.trainable_variables)

                    optimizer.apply_gradients(zip(grad_actor, self.actor.trainable_variables))

            for var1, var2 in zip(target_actor.trainable_variables,self.actor.trainable_variables):
                var1.assign(polyak * var1 + (1-polyak)* var2)
            for var1, var2 in zip(target_critic.trainable_variables,self.critic.trainable_variables):
                var1.assign(polyak * var1 + (1-polyak)* var2)
            print(sum_reward,'...' ,noise)
        return commulative_reward_history

    @tf.function
    def q_value(self,model,state, action):
        concated = tf.concat([state, action], axis=-1)
        return model(concated)

    def control(self, state):
        action =  np.clip(self.actor(state),self.min_value, self.max_value)
        return action





    

        

