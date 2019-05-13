import gym
import time
import random
import numpy as np
from collections import deque
import tensorflow as tf
# import matplotlib.pyplot as plt
# %matplotlib inline
from tf_dqn_agent import Agent

def main():
    env = gym.make('LunarLander-v2')
    # env.seed(0)
    print('State shape: ', env.observation_space.shape)
    print('Number of actions: ', env.action_space.n)

    state = env.reset()
    print('state example: ', state)

    agent = Agent(state_size=env.observation_space.shape[0], action_size=env.action_space.n)

    episodes = 5000
    scores_window = deque(maxlen=100)
    scores = []
    for e in range(episodes + 1):
        state = env.reset()
        state = np.reshape(state, [1, env.observation_space.shape[0]])
        score = 0

        for t in range(1000):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, 8])
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                print("episode: {}/{}, score: {}, after time: {}".format(e, episodes, score, t))
                break
            
            if ((t+1)% 4) == 0:
                if agent.memory.__len__()>=64:
                    agent.replay()

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        scores.append(score)
        scores_window.append(score)
        if (e)%100 == 0:
            mean_score = np.mean(scores_window)
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(e, mean_score))
            if mean_score >= 200.0:
                agent.qnetwork.save_weights('weights_{}.h5'.format(e))
                break
        
        if e%200 == 0:
            agent.qnetwork.save_weights('weights_{}.h5'.format(e))

if __name__ == "__main__":
    main()