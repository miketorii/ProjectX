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
        
if __name__ == "__main__":
    print("--------start----------")
    nonstat = NonStatBadit()
    ret = nonstat.play(1)
    print(ret)
    print("--------end----------")    
    
