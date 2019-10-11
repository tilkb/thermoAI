import tensorflow as tf
from tqdm import tqdm
import numpy as np

class iLQR:
    def __init__(self, feature_size, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value
        self.feature_size = feature_size
        self.transition_model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation=tf.nn.relu, input_shape=(feature_size+1,)),
            tf.keras.layers.Dense(128, activation=tf.nn.tanh),
            tf.keras.layers.Dense(feature_size)
        ], name="Environment_model")
        self.cost_model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation=tf.nn.relu, input_shape=(feature_size+1,)),
            tf.keras.layers.Dense(128, activation=tf.nn.tanh),
            tf.keras.layers.Dense(1)
        ], name="Cost_model")

        self.transition_model.compile(loss='mse', optimizer='adam')
        self.cost_model.compile(loss='mse', optimizer='adam')

        self.collected_data = [[],[],[]]

    def backward(self, x_seq, u_seq):
        k_seq = []
        kk_seq = []
        v_x = [np.zeros((self.feature_size, 1)) for _ in range(len(x_seq) + 1)]
        v_xx = [np.zeros((self.feature_size, self.feature_size)) for _ in range(len(x_seq) + 1)]

        for t in range(len(x_seq)-1,-1,-1):
            f,c , F_x,c_x,C_xx = self.get_derivatives(x_seq[t],u_seq[t])
            q = c_x + np.matmul(F_x.T,v_x[t+1]) + np.matmul(F_x.T,np.matmul(v_xx[t+1],f)).reshape(-1,1)
            Q = C_xx + np.matmul(np.matmul(F_x.T,v_xx[t+1]),F_x)
            q_uu = Q[-1:,-1:]
            q_u = q[-1:,:]
            q_ux = Q[-1:,:-1]
            inv_q_uu = 1.0/q_uu
            k=-np.matmul(inv_q_uu,q_u)
            kk=-np.matmul(inv_q_uu,q_ux)
            #-------------

            v_x[t] = q[:-1] - np.matmul(np.matmul(q_u, inv_q_uu), q_ux).T
            v_xx[t] = Q[:-1,:-1] + np.matmul(q_ux.T, kk)


            k_seq.append(k)
            kk_seq.append(kk)

        k_seq.reverse()
        kk_seq.reverse()
        return k_seq, kk_seq
    
    def forward(self,x_seq, u_seq, k_seq, kk_seq, alpha=1.0):
        u_seq_hat = np.array(u_seq)
        x_seq_hat = np.array(x_seq)
        for t in range(len(u_seq)-1):
            control_dif = k_seq[t]+ np.matmul(kk_seq[t],(x_seq_hat[t]-x_seq[t]))
            u_seq_hat[t] = np.clip(u_seq[t] + alpha * control_dif, self.min_value, self.max_value)
            x = tf.reshape(tf.constant(x_seq_hat[t], dtype=tf.float32),[1,-1])
            u = tf.reshape(tf.constant(u_seq_hat[t], dtype=tf.float32),[1,-1])
            input = tf.concat([x,u], axis=1)
            x_seq_hat[t+1] = self.transition_model(input).numpy()
        return x_seq_hat, u_seq_hat

    def get_derivatives(self,x,u):
        x = tf.constant(x, dtype=tf.float32)
        u = tf.constant(u, dtype=tf.float32)
        input = tf.reshape(tf.concat([x, u], axis=-1),(1,-1))
        with tf.GradientTape(persistent=True) as tape0:
            tape0.watch(input)
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(input)
                cost = tf.reshape(self.cost_model(input),[1])
                next_state = tf.reshape(self.transition_model(input), [self.feature_size])
            c = tape.gradient(cost, input)
            F = tape.jacobian(next_state, input)
        c_second_derivative = tape0.jacobian(c,input)
        del tape
        del tape0
        return next_state.numpy(), cost.numpy(),F.numpy().reshape(self.feature_size,self.feature_size+1),c.numpy().reshape(-1,1), c_second_derivative.numpy().reshape(self.feature_size+1,self.feature_size+1)

    def run_episode(self,sim, time_horizon=5):
        sim.reset()
        u_seq = [np.zeros(1) for t in range(time_horizon)]
        x_seq = [sim.get_concated_features() for t in range(time_horizon)]
        done=False
        while not(done):
            for i in range(5):
                k_seq, kk_seq = self.backward(x_seq,u_seq)
                x_seq,kk_seq = self.forward(x_seq,u_seq, k_seq, kk_seq)
            state = tf.constant(sim.get_concated_features(), dtype=tf.float32)

            done, reward, _ = sim.step(u_seq[0][0])
            x_seq[0] = sim.get_concated_features()
            #store collected data
            next_state = tf.constant(sim.get_concated_features(), dtype=tf.float32)
            reward = tf.constant([reward], dtype=tf.float32)
            self.collected_data[0].append(tf.reshape(tf.concat([tf.reshape(state, (1, -1)), tf.reshape(tf.constant(u_seq[0][0],dtype=tf.float32), (1, -1))], axis=-1), [-1]))
            self.collected_data[1].append(next_state)
            self.collected_data[2].append(reward)
            print('.')

    def fit_model(self, epoch=1):
        state_action = np.array(self.collected_data[0])
        next_state = np.array(self.collected_data[1])
        cost = np.array(self.collected_data[2])
        transition_data = tf.data.Dataset.from_tensor_slices((state_action, next_state)).batch(64)
        cost_data = tf.data.Dataset.from_tensor_slices((state_action, cost)).batch(64)
        self.transition_model.fit(transition_data, epochs=epoch)
        self.cost_model.fit(cost_data, epochs=epoch)


    def train(self, simulator,warmup_time=100, episode=10, time_horizon=50):
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

        self.fit_model(epoch=3)
        for iteration in tqdm(range(episode)):
            self.v = [0.0 for _ in range(time_horizon + 1)]

            self.run_episode(simulator, time_horizon)
            self.fit_model()



    