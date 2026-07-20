import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, arms=10):
        self.rates=np.random.rand(arms)
#        print(self.rates)

    def play(self, arm):
        rate = self.rates[arm]
        if rate > np.random.rand():
            return 1
        else:
            return 0
        
class Agent:
    def __init__(self, epsilon, action_size=10):
        self.epsilon = epsilon
        self.Qs = np.zeros(action_size)
        self.ns = np.zeros(action_size)

    def update(self, action, reward):
        self.ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]

    def get_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))
        else:
            return np.argmax(self.Qs)
        
        
        
if __name__ == "__main__":
    print("------start-------")

    steps = 1000
#    steps = 3
    epsilon = 0.1

    bandit = Bandit()
    agent = Agent(epsilon)
    
    total_reward = 0
    total_rewards = []
    rates = []

    for step in range(steps):
        action = agent.get_action()
#        print(action)
        reward = bandit.play(action)
#        print(reward)
        agent.update(action, reward)
        total_reward += reward

        total_rewards.append(total_reward)
        rates.append( total_reward/(step+1) )
        
    print(total_reward)
    
    plt.xlabel("Total reward")
    plt.ylabel("Steps")
    plt.plot(total_rewards)
    plt.savefig("rewards.png")
    
    plt.xlabel("Rates")
    plt.ylabel("Steps")
    plt.plot(rates)
    plt.savefig("rates.png")
    
    print("------end-------")
