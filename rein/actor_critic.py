import numpy as np
import gymnasium as gym
from dezero import Model
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L

####################################################
# Policy class
#
####################################################
class PolicyNet(Model):
    def __init__(self, action_size=2):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        x = F.softmax(x)
        return x

####################################################
#
#
class ValueNet(Model):
    def __init__(self):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x
    
####################################################
# Agenda class
#
####################################################
class Agent:
    def __init__(self):
        self.gamma = 0.98
        self.lr_pi = 0.0002
        self.lr_v = 0.0005        
        self.action_size = 2

        self.pi = PolicyNet()
        self.v = ValueNet()
        self.optimizer_pi = optimizers.Adam(self.lr_pi).setup(self.pi)
        self.optimizer_v = optimizers.Adam(self.lr_v).setup(self.v)


    def get_action(self, state):
        state = state[np.newaxis, :]
        probs = self.pi(state)
        probs = probs[0]
        action = np.random.choice(len(probs), p=probs.data)
        return action, probs[action]
    
    def update(self, state, action_prob, reward, next_state, done):
        state = state[np.newaxis, :]
        next_state = next_state[np.newaxis, :]

        target = reward + self.gamma + self.v(next_state) * (1 - done)
        target.unchain()
        v = self.v(state)
        loss_v = F.mean_squared_error(v, target)

        delta = target - v
        delta.unchain()
        loss_pi = -F.log(action_prob) * delta

        self.v.cleargrads()
        self.pi.cleargrads()        
        loss_v.backward()
        loss_pi.backward()
        self.optimizer_v.update()
        self.optimizer_pi.update()
        
####################################################
# main
#
if __name__ == '__main__':
    print("---------------start---------------")

    episodes = 100
    #episodes = 3000    
    env = gym.make('CartPole-v1', render_mode='human')
    agent = Agent()
    reward_history = []
    
    for episode in range(episodes):
        state, info = env.reset()

        done = False
        truncated = False

        total_reward = 0

        while not (done or truncated):
            action, prob = agent.get_action(state) #np.random.choice([0,1])
            next_state, reward, done, truncated, info = env.step(action)

            agent.update(state, prob, reward, next_state, done)
            
            state = next_state
            total_reward += reward

        reward_history.append(total_reward)
        if episode % 100 == 0:
            print("episode:{}, total reward: {:.1f}".format(episode, total_reward))

        #env.close()
