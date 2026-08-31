import numpy as np
import gymnasium as gym
from dezero import Model
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L

####################################################
#
#
class Policy(Model):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x))
        return x

####################################################
#
#
class Agent:
    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.0002
        self.action_size = 2

        self.memory = []
        self.pi = Policy(self.action_size)
        self.optimizer = optimizers.Adam(self.lr)
        self.optimizer.setup(self.pi)

    def add(self, reward, prob):
        data = (reward, prob)
        self.memory.append(data)
        
####################################################
#
#
if __name__ == '__main__':
    print("---------------start---------------")

    episodes = 3 #3000
    env = gym.make('CartPole-v1', render_mode='human')
    agent = Agent()
    reward_history = []
    
    for episode in range(episodes):
        state, info = env.reset()

        done = False
        truncated = False

        while not (done or truncated):
            action = np.random.choice([0,1])
            next_sate, reward, done, truncated, info = env.step(action)

        #env.close()
