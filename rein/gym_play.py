import numpy as np
import gymnasium as gym

env = gym.make('CartPole-v1', render_mode='human')
state, info = env.reset()

done = False
truncated = False

while not (done or truncated):
    #env.render()
    action = np.random.choice([0,1])
    next_sate, reward, done, truncated, info = env.step(action)
env.close()
