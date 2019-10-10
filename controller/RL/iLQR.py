import tensorflow as tf
from tqdm import tqdm
import numpy as np

class iLQR:
    def __init__(self, feature_size, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value
        self.transition_model = tf.keras.Sequential([
            tf.keras.layers.Dense(200, activation=tf.nn.relu, input_shape=(feature_size+1,)),
            tf.keras.layers.Dense(200, activation=tf.nn.relu),
            tf.keras.layers.Dense(feature_size)
        ], name="Environment_model")
        self.cost_model = tf.keras.Sequential([
            tf.keras.layers.Dense(200, activation=tf.nn.relu, input_shape=(feature_size+1,)),
            tf.keras.layers.Dense(200, activation=tf.nn.relu),
            tf.keras.layers.Dense(1)
        ], name="Cost_model")
        self.transition_model.compile(loss='mse', optimizer='adam')
        self.cost_model.compile(loss='mse', optimizer='adam')

        self.collected_data = [[],[],[]]

    def backward(self, x_seq, u_seq):
        k_seq = []
        kk_seq = []
        v = [0.0 for _ in range(len(x_seq)+ 1)]
        v_x = [0.0 for _ in range(len(x_seq) + 1)]
        v_xx = [0.0 for _ in range(len(x_seq) + 1)]
        #v[-1] =
        #v_x[-1] =
        #v_xx[-1] =

        for t in range(len(x_seq)-1,-1,-1):
            f,c , F_x,c_x,C_xx = self.get_derivatives(x_seq[t],u_seq[t])
            q = c_x + np.matmul(F_x.T,v_x[t+1])
            Q = C_xx + np.matmul(np.matmul(F_x.T,v_xx[t+1]),F_x) + np.dot(v_x[t+1],f_xx)


            q_uu = Q[-1,-1]
            q_u = q[-1]
            q_ux = Q[-1,:-1]
            inv_q_uu = np.linalg.inv(q_uu)
            k=-np.matmul(inv_q_uu,q_u)
            kk=-np.matmul(inv_q_uu,q_ux)
            dv = 0.5* np.matmul(q_u,k)
            v[t] += dv
            v_x[t] = q[:-1] - np.matmul(np.matmul(q_u, inv_q_uu), q_ux)
            v_xx[t] = Q[:-1,:-1] + np.matmul(q_ux.T, kk)

            k_seq.append(k)
            kk_seq.append(kk)

        k_seq.reverse()
        kk_seq.reverse()
        return k_seq, kk_seq
    
    def forward(self,x_seq, u_seq, k_seq, kk_seq):
        u_seq_hat = np.array(u_seq)
        x_seq_hat = np.array(x_seq)
        for t in range(len(u_seq)):
            control = k_seq[t]+ np.matmul(kk_seq[t],(x_seq_hat[t]-x_seq[t]))
            u_seq_hat[t] = np.clip(u_seq[t] + control, self.min_value, self.max_value)
            x_seq_hat[t+1] = self.transition_model(x_seq_hat[t],u_seq_hat[t])
        return x_seq_hat, u_seq_hat

    def get_derivatives(self,x,u):
        with tf.GradientTape(persistent=True) as tape0:
            with tf.GradientTape(persistent=True) as tape:
                cost = self.cost_model(x)
                next_state = self.transition_model(tf.concat([x, u], axis=-1))
            c = tape.gradient(cost, [x,u])
            F = tape.gradient(next_state, [x,u])
        c_second_derivative = tape0.gradient(c,[x,u])
        return next_state, cost,F,c, c_second_derivative

    def run_episode(self,sim, time_horizon=50):
        sim.reset()
        u_seq = [np.zeros(1) for t in range(time_horizon)]
        x_seq = [sim.get_concated_features() for t in range(time_horizon)]
        done=False
        while not(done):
            for i in range(5):
                k_seq, kk_seq = self.backward(x_seq,u_seq)
                x_seq,kk_seq = self.forward(x_seq,u_seq, k_seq, kk_seq)
            state = tf.constant(sim.get_concated_features(), dtype=tf.float32)
            done, reward, _ = sim.step(u_seq[0])
            x_seq[0] = sim.get_concated_features()
            #store collected data
            next_state = tf.constant(sim.get_concated_features(), dtype=tf.float32)
            reward = tf.constant([reward], dtype=tf.float32)
            self.collected_data[0].append(tf.reshape(tf.concat([tf.reshape(state, (1, -1)), tf.reshape(act, (1, -1))], axis=-1), [-1]))
            self.collected_data[1].append(next_state)
            self.collected_data[2].append(reward)

    def fit_model(self, epoch=1):
        state_action = np.array(self.collected_data[0])
        next_state = np.array(self.collected_data[1])
        cost = np.array(self.collected_data[2])
        transition_data = tf.data.Dataset.from_tensor_slices((state_action, next_state)).batch(64)
        cost_data = tf.data.Dataset.from_tensor_slices((state_action, cost)).batch(64)
        self.transition_model.fit(transition_data, epochs=epoch)
        self.cost_model.fit(cost_data, epochs=epoch)


    def train(self, simulator,warmup_time=100, episode=10):
        #collect transitions to initialize the model
        for iteration in range(warmup_time):
            simulator.reset()
            done = False
            while not (done):
                random_act = np.random.uniform(self.min_value, self.max_value)
                state = tf.constant(simulator.get_concated_features(), dtype=tf.float32)
                done, reward, _ = simulator.step(random_act)
                next_state = tf.constant(simulator.get_concated_features(), dtype=tf.float32)
                reward = tf.constant([reward], dtype=tf.float32)
                act = tf.constant(random_act, dtype=tf.float32)
                self.collected_data[0].append(tf.reshape(tf.concat([tf.reshape(state,(1,-1)),tf.reshape(act,(1,-1))], axis=-1),[-1]))
                self.collected_data[1].append(next_state)
                self.collected_data[2].append(reward)

        self.fit_model(epoch=10)
        for iteration in tqdm(range(episode)):
            self.run_episode(simulator)
            self.fit_model()



    