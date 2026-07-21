import numpy as np
import matplotlib.pyplot as plt
from bandit import Agent

class NonStatBadit:
    def __init__(self, arms=10):
        self.arms = arms
        self.rates = np.random.rand(arms)

    def play(self, arm):
        rate = self.rates[arm]
        self.rates += 0.1 * np.random.randn(self.arms)
        if rate > np.random.rand():
            return 1
        else:
            return 0

class AlphaAgent:
    def __init__(self, epsilon, alpha, actions=10):
        self.epsilon = epsilon
        self.Qs = np.zeros(actions)
        self.alpha = alpha

    def update(self, action, reward):
        self.Qs[action] += (reward - self.Qs[action]) * self.alpha

    def get_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))
        return np.argmax(self.Qs)
    
        
if __name__ == "__main__":
    print("--------start----------")
    runs = 200
    steps = 1000
    epsilon = 0.1
    alpha = 0.8
    agent_types = ['simple average','alpha const update']
    results = {}
    
    nonstat = NonStatBadit()
    ret = nonstat.play(1)
    print(ret)

    agent = AlphaAgent(epsilon, alpha)
    actionnum = agent.get_action()
    print(actionnum)
    reward = 5
    agent.update(actionnum, reward)
    
    print("--------end----------")    
    
