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
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0005  # or 5e-4
        self.qnetwork = TfQNetwork(self.action_size)
        self.qnetwork.compile(optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate),
                           loss='mse')
        self.target_network = TfQNetwork(self.action_size)
        self.target_network.compile(optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate),
                              loss='mse')
        self.update_target_network()
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size)

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

    def replay(self):
        states, actions, rewards, next_states, dones = self.memory.sample()
        outputs = self.target_network.predict(next_states)
        targets = rewards + (self.gamma * np.reshape(np.amax(outputs, axis=1), [64, 1])*(1.0 - dones))
        predicts = self.qnetwork.predict(states)
        for i in range(predicts.shape[0]):
            predicts[i][actions[i]] = targets[i]
                
        self.qnetwork.fit(states, predicts, epochs=1, verbose=0)

        self.update_target_network()
    
    def replay_with_loop_style(self):
        experience = self.memory.sample_for_loop_style()
        batch_state = []
        batch_target = []
        for state, action, reward, next_state, done in experience:
            target = self.qnetwork.predict(next_state)
            target[0][action] = reward if done else reward + self.gamma * np.max(target[0])
            predict = self.qnetwork.predict(state)
            batch_state.append(state[0])
            batch_target.append(target[0])
        
        self.qnetwork.fit(np.array(batch_state), np.array(batch_target), epochs=1, verbose=0)
        self.update_target_network()

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, self.action_size)
        act_values = self.qnetwork.predict(state)
        return np.argmax(act_values[0])

    def update_target_network(self):
        
        self.target_network.set_weights(self.qnetwork.get_weights())
