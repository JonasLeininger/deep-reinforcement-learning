import numpy as np
import tensorflow as tf
import random
from collections import namedtuple, deque



class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = tf.Variable(np.vstack([e.state for e in experiences if e is not None]), dtype=tf.float32)
        actions = tf.Variable(np.vstack([e.action for e in experiences if e is not None]), dtype=tf.float32)
        rewards = tf.Variable(np.vstack([e.reward for e in experiences if e is not None]), dtype=tf.float32)
        next_states = tf.Variable(np.vstack([e.next_state for e in experiences if e is not None]), dtype=tf.float32)
        dones = tf.Variable(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8), dtype=tf.float32)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

if __name__=='__main__':
    BUFFER_SIZE = int(1e5)  # replay buffer size
    BATCH_SIZE = 64         # minibatch size
    GAMMA = 0.99            # discount factor
    TAU = 1e-3              # for soft update of target parameters
    LR = 5e-4               # learning rate 
    UPDATE_EVERY = 4        # how often to update the network
    replayBuff = ReplayBuffer(4, BUFFER_SIZE, 2, 23)
    print(replayBuff.memory)
    replayBuff.add(1,1,1,2,1)
    replayBuff.add(4,2,1,2,1)
    replayBuff.add(4,2,1,2,1)
    replayBuff.add(4,3,1,2,1)
    print(replayBuff.memory)
    st, act, r, nst, d = replayBuff.sample()
    print(st, act)