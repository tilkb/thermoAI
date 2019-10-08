cdimport tensorflow as tf
from tqdm import tqdm


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

        self.transition_data=[]
        self.cost_pred_data=[]

    def backward(self, x_seq, u_seq):
        k_seq = []
        kk_seq = []
        for t in range(len(x_seq)-1,-1,-1):
            F,c,C = self.get_derivatives(x_seq[t],u_seq[t])
            Q=C+F*V*F
            q = c+F*V
            #TODO...

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
        C = tape0.gradient(c,[x,u])
        return F,c C

    def run_episode(self,sim, time_horizon=50):
        sim.reset()
        u_seq = [np.zeros(1) for t in range(time_horizon)]
        x_seq = [sim.get_concated_features() for t in range(time_horizon)]
        done=True
        while not(done):
            for i in range(5):
                k_seq, kk_seq = self.backward(x_seq,u_seq)
                x_seq,kk_seq = self.forward(x_seq,u_seq, k_seq, kk_seq)
            done, reward, _ = sim.step(u_seq[0])
            state = sim.get_concated_features()
            x_seq[0] = state

    def fit_model(self, epoch=1):
        pass

    def train(self, simulator,warmup_time=30):
        #collect transitions to initialize the model
        for iteration in range(warmup_time):
            sim.reset()
            done = True
            while not (done):
                random_act = np.random.uniform(self.min_value, self.max_value)
                state = sim.get_concated_features()
                done, reward, _ = simulator.step(random_act)
                next_state = sim.get_concated_features()

        self.fit_model(epoch=10)
        for iteration in range(episode):
            self.run_episode(simulator)
            self.fit_model()
    