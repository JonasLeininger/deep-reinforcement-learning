import numpy as np
import tensorflow as tf

from replay_buffer import ReplayBuffer
from tf_model import TfQNetwork

TAU = 1e-3  # for soft update of target parameters

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.buffer_size = int(1e5)
        self.batch_size = 64
        self.tau = 1e-3
        self.update_every = 4
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001  # or 5e-4
        self.qnetwork = TfQNetwork(self.action_size)
        self.qnetwork.compile(optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate),
                           loss='mse')
        self.target_network = TfQNetwork(self.action_size, name="target_network")
        self.target_network.compile(optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate),
                              loss='mse')
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size)

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

    def replay(self):
        experience = self.memory.sample()
        for state, action, reward, next_state, done in experience:
            target = reward
            if not done:
                target = reward + self.gamma*np.amax(self.qnetwork.predict(next_state)[0])
            future_target = self.qnetwork.predict(state)
            future_target[0][action] = target
            self.qnetwork.fit(state, future_target, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, self.action_size)
        act_values = self.qnetwork.predict(state)
        return np.argmax(act_values[0])

    def update_target_network(self):
        self.target_network.set_weights(self.qnetwork.get_weights())
