import numpy as np
from dezero import Model
from dezero import optimizers
import dezero.layers as L
import dezero.functions as F
from gridworld import GridWorld

#####################################################
#
#
def one_hot(state):
    HIGHT, WEDTH = 3, 4
    vec = np.zeros(HEIGHT * WIDTH, dtype=np.float32)
    y, x = state
    idx = WIDTH * y + x
    vec[idx] = 1.0
    return vec[np.newaxis, :]

#####################################################
#
#
class QNet(Model):
    def __init__(self):
        super().__init__()
        self.l1 = L.Linear(100)
        self.l2 = L.Linear(4)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x

#####################################################
#
#
class QLearningAgent:
    def __init__(self):
        self.gamma = 0.9
        self.lr = 0.01
        self.epsilon = 0.1
        self.action_size = 4

        self.qnet = QNet()
        self.optimizer = optimizers.SGD(self.lr)
        self.optimizer.setup(self.qnet)

    def get_action(self, start_vec):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            qs = self.qnet(state_vec)
            return qs.data.argmax()

    def update(self, state, action, reward, next_state, done):
        if done:
            next_q = np.zeros(1)
        else:
            next_qs = self.qnet(next_state)
            next_q = next_qs.max(axis=1)
            next_q.unchain()

        target = self.gamma * next_q + reward
        qs = self.qnet(state)
        q = qs[:, action]
        loss = F.mean_squared_error(target, q)

        self.qnet.cleargrads()
        loss.backward()
        self.optimizer.update()

        return loss.data

if __name__ == "__main__":
    print('-----------start-------------')

    np.random.seed(0)
    x = np.random.rand(100,1)
    y = np.sin(2 * np.pi * x) + np.random.rand(100,1)

    lr = 0.2
    iters = 10000


    model = TwoLayerNet(10,1)
    optimizer = optimizers.SGD(lr)
    optimizer.setup(model)

    for i in range(iters):
        y_pred = model(x)
        loss = F.mean_squared_error(y, y_pred)

        model.cleargrads()
        loss.backward()

        optimizer.update()
        if i % 1000 == 0:
            print(loss.data)
            
    t = np.arange(0,10,1)[:,np.newaxis]
    y_pred = model(t)
    print(y_pred)

    print('-----------end-------------')



