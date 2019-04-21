import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6, alpha=0.1, epsilon=1., gamma=1):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def select_action(self, state, i_episode):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        self.epsilon = 1. / i_episode
        return self.epsilon_greedy(state)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """

        self.update_q_learning(state, action, reward, next_state)

    def epsilon_greedy(self, state):
        if np.random.random() > self.epsilon:
            return np.argmax(self.Q[state])
        else: 
            return np.random.choice(np.arange(self.nA))
    
    def update_q_learning(self, state, action, reward, next_state=None):
        current = self.Q[state][action]
        q_sa_next = np.max(self.Q[next_state]) if next_state is not None else 0
        target = reward + (self.gamma * q_sa_next)
        sum_error = (target + current)
        update_error = self.alpha * sum_error
        self.Q[state][action] = current + update_error
        