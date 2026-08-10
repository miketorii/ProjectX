import numpy as np
from collections import defaultdict
from gridworld import GridWorld

def greedy_probs(Q, state, epsilon=0, action_size=4):
    qs = [ Q[(state, action)] for action in range(action_size)]
    max_action = np.argmax(qs)

    base_prob = epsilon / action_size
    action_probs = {action: base_prob for action in range(action_size)}
    action_probs[max_action] += (1 - epsilon)    
    return action_probs

class TdAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.01
        self.action_size = 4

        random_actions = {0:0.25, 1:0.25, 2:0.25, 3:0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.V = defaultdict(lambda: 0)

    def get_action(self, state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)
    
    def eval(self, state, reward, next_state, done):
        next_V = 0 if done else self.V[next_state]

        target = self.gamma * next_V + reward
        self.V[state] += (target - self.V[state]) * self.alpha

if __name__ == '__main__':
    print("--------Start TD eval-----------")
    env = GridWorld()
    agent = TdAgent()

    episodes = 10000
    for episode in range(episodes):
        state = env.reset()

        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)

            agent.eval(state, reward, next_state, done)
            if done:
                break

            state = next_state
            
    print(agent.V)
    
    print("--------End-----------")    
