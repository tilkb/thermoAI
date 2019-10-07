import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm
from controller.RL.utils.common import ReplayMemory, Transition

class SAC:
    def __init__(self, feature_size, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value
        self.policy = tf.keras.Sequential([
            tf.keras.layers.Dense(200, activation=tf.nn.relu, input_shape=(feature_size,)),
            tf.keras.layers.Dense(200, activation=tf.nn.relu),
            tf.keras.layers.Dense(1)
        ], name="Policy")
        self.q1 = tf.keras.Sequential([
            tf.keras.layers.Dense(200, activation=tf.nn.relu, input_shape=(feature_size+1,)),
            tf.keras.layers.Dense(200, activation=tf.nn.relu),
            tf.keras.layers.Dense(1)
        ], name="Q1_network")
        self.q2 = tf.keras.Sequential([
            tf.keras.layers.Dense(200, activation=tf.nn.relu, input_shape=(feature_size+1,)),
            tf.keras.layers.Dense(200, activation=tf.nn.relu),
            tf.keras.layers.Dense(1)
        ], name="Q2_network")
        self.value = tf.keras.Sequential([
            tf.keras.layers.Dense(200, activation=tf.nn.relu, input_shape=(feature_size,)),
            tf.keras.layers.Dense(200, activation=tf.nn.relu),
            tf.keras.layers.Dense(1)
        ], name="Value_network")
    
    def train(self,simulator,episode=1000, batch_size=64, gamma=0.95,init_step=0,noise_decay=0.98, alpha=0.):
        polyak = tf.constant([0.95])
        noise = 0.1
        gamma = tf.constant([gamma])
        optimizer = tf.keras.optimizers.Adam()
        replay_memory = ReplayMemory(10000)
        target_value_network = tf.keras.models.clone_model(self.value)
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
                action_dist = tfp.distributions.Normal(loc=self.policy(state), scale=noise)
                action = tf.clip_by_value(action_dist.sample(1),self.min_value, self.max_value).numpy().reshape([1])
                done, reward, _ = simulator.step(action[0])
                sum_reward += reward
                next_state = tf.constant(simulator.get_concated_features(), dtype=tf.float32)
                if not(done):
                    replay_memory.push(state,tf.constant(action, dtype=tf.float32),next_state, tf.constant(reward, dtype=tf.float32))

                #actual training
                if len(replay_memory)>=batch_size:
                    transitions = replay_memory.sample(batch_size)
                    batch = Transition(*zip(*transitions))
                    state = tf.reshape(tf.stack(batch.state,axis=0),(batch_size,-1))
                    action = tf.reshape(tf.stack(batch.action,axis=0),(batch_size,-1))
                    next_state = tf.reshape(tf.stack(batch.next_state,axis=0),(batch_size,-1))
                    reward = tf.reshape(tf.stack(batch.reward,axis=0),(batch_size,-1))
                    with tf.GradientTape(persistent=True) as tape:
                        y_q = reward+gamma*target_value_network(next_state)
                        sample_dist = tfp.distributions.Normal(loc=self.policy(state), scale=noise)
                        sampled_action = tf.clip_by_value(sample_dist.sample(1),self.min_value, self.max_value)
                        print(sampled_action)
                        log_prob = sample_dist.log_prob(sampled_action)
                        sampled_action = tf.reshape(sampled_action,(-1,1))
                        y_v = tf.minimum(self.q_value(self.q1,state,sampled_action),self.q_value(self.q2,state,sampled_action)) - alpha * log_prob
                        loss1 = tf.math.square(y_q-self.q_value(self.q1,state,action))
                        loss2 = tf.math.square(y_q-self.q_value(self.q2,state,action))
                        loss_value = tf.math.square(y_v-self.value(state))
                        loss_policy = -(self.q_value(self.q1,state,sampled_action)-alpha*log_prob)
                    grad1 = tape.gradient(loss1, self.q1.trainable_variables)
                    optimizer.apply_gradients(zip(grad1, self.q1.trainable_variables))
                    grad2 = tape.gradient(loss2, self.q2.trainable_variables)
                    optimizer.apply_gradients(zip(grad2, self.q2.trainable_variables))
                    grad_value = tape.gradient(loss_value, self.value.trainable_variables)
                    optimizer.apply_gradients(zip(grad_value, self.value.trainable_variables))
                    grad_policy = tape.gradient(loss_policy, self.policy.trainable_variables)
                    optimizer.apply_gradients(zip(grad_policy, self.policy.trainable_variables))
                    
                    #update target network
                    for var1, var2 in zip(target_value_network.trainable_variables,self.value.trainable_variables):
                        var1.assign(polyak * var1 + (1-polyak)* var2)
            print(sum_reward)
    
    
    #@tf.function
    def q_value(self,model,state, action):
        concated = tf.concat([state, action], axis=-1)
        return model(concated)
        
