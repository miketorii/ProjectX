import numpy as np

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
        

if __name__ == "__main__":
    print("------start-------")

    steps = 1000
    epsilon = 0.1

    bandit = Bandit()
    r = bandit.play(1)
    

    print(r)
    
    print("------end-------")
