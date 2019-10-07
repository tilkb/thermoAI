import tensorflow as tf
from tqdm import tqdm


class iLQR:
    def __init__(self, feature_size, min_value, max_value):
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

    def backward(self, x_seq, u_seq):
        for t in range(len(x_seq)-1,-1,-1):
            F,c,C = self.get_derivatives(x_seq[t],u_seq[t])
            Q=C+F**F
            q = c+F*V
    
    def forward(self,x_seq, u_seq, k_seq, kk_seq):
        pass

    def get_derivatives(x,u):
        with tf.GradientTape(persistent=True) as tape0:
            with tf.GradientTape(persistent=True) as tape:
                cost = self.cost_model(x)
                next_state = self.transition_model(tf.concat([x, u], axis=-1))
            c = tape.gradient(cost, [x,u])
            F = tape.gradient(next_state, [x,u])
        C = tape0.gradient(c,[x,u])
        return F,c C

    def train(self, simulator,):
        #collect transitions to initialize the model
        for iteration in range(warmup):

        for iteration in range(episode):
        
    