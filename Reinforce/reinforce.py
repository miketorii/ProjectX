import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(5,5))
ax = plt.gca()

plt.plot([1,1],[0,1],color='red', linewidth=2)
plt.plot([1,2],[2,2],color='red', linewidth=2)
plt.plot([2,2],[2,1],color='red', linewidth=2)
plt.plot([2,3],[1,1],color='red', linewidth=2)

plt.text(0.5, 2.5, 's0', size=14, ha='center')
plt.text(1.5, 2.5, 's1', size=14, ha='center')
plt.text(2.5, 2.5, 's2', size=14, ha='center')
plt.text(0.5, 1.5, 's3', size=14, ha='center')
plt.text(1.5, 1.5, 's4', size=14, ha='center')
plt.text(2.5, 1.5, 's5', size=14, ha='center')
plt.text(0.5, 0.5, 's6', size=14, ha='center')
plt.text(1.5, 0.5, 's7', size=14, ha='center')
plt.text(2.5, 0.5, 's8', size=14, ha='center')
plt.text(0.5, 2.3, 'START', ha='center')
plt.text(2.5, 0.3, 'GOAL', ha='center')

ax.set_xlim(0,3)
ax.set_ylim(0,3)
plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')

line, = ax.plot([0.5], [2.5], marker='o', color='g', markersize=60)

theta_0 = np.array([[np.nan, 1, 1, np.nan],
                    [np.nan, 1, np.nan, 1],
                    [np.nan, np.nan, 1, 1],
                    [1, 1, 1, np.nan],
                    [np.nan, np.nan, 1, 1],
                    [1, np.nan, np.nan, np.nan],
                    [1, np.nan, np.nan, np.nan],
                    [1, 1, np.nan, np.nan]])

def simple_convert_into_pi_from_theta(theta):
  [m,n] = theta.shape
  pi = np.zeros((m,n))
  for i in range(0,m):
    pi[i, :] = theta[i, :] / np.nansum(theta[i, :])

  pi = np.nan_to_num(pi)

  return pi

pi_0 = simple_convert_into_pi_from_theta(theta_0)
pi_0

def get_next_s(pi, s):
  direction = ["up", "right", "down", "left"]

  next_direction = np.random.choice(direction, 1, p=pi[s, :])
  #print(next_direction)

  if next_direction == "up":
    s_next = s - 3
  elif next_direction == "right":
    s_next = s + 1
  elif next_direction == "down":
    s_next = s + 3
  elif next_direction == "left":
    s_next = s - 1

  return s_next

def goal_maze(pi):
  s = 0
  state_history = [0]

  while(1):
    next_s = get_next_s(pi, s)
    state_history.append(next_s)
    if next_s == 8:
      break
    else:
      s = next_s

  return state_history

state_history = goal_maze(pi_0)
print(state_history)

from matplotlib import animation
from IPython.display import HTML

def init():
  line.set_data([],[])
  return (line,)

def animate(i):
  state = state_history[i]
  x = (state % 3) + 0.5
  y = 2.5 - int(state /3)
  line.set_data(x, y)
  return (line,)

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(state_history), interval=200, repeat=False)

HTML(anim.to_jshtml())

def softmax_convert_into_pi_from_theta(theta):
  beta = 1.0
  [m,n] = theta.shape
  pi = np.zeros((m,n))

  exp_theta = np.exp( beta * theta )

  for i in range(0,m):
    pi[i, :] = exp_theta[i, :] / np.nansum(exp_theta[i, :])

  pi = np.nan_to_num(pi)

  return pi

pi_0 = softmax_convert_into_pi_from_theta(theta_0)
pi_0

def get_action_and_next_s(pi, s):
  direction = ["up", "right", "down", "left"]

  next_direction = np.random.choice(direction, 1, p=pi[s, :])
  #print(next_direction)

  if next_direction == "up":
    action = 0
    s_next = s - 3
  elif next_direction == "right":
    action = 1
    s_next = s + 1
  elif next_direction == "down":
    action = 2
    s_next = s + 3
  elif next_direction == "left":
    action = 3
    s_next = s - 1

  return [action, s_next]

def goal_maze_ret_s_a(pi):
  s = 0
  s_a_history = [[0, np.nan]]

  while(1):
    action, next_s = get_action_and_next_s(pi, s)
    s_a_history[-1][1] = action

    s_a_history.append([next_s, np.nan])
    if next_s == 8:
      break
    else:
      s = next_s

  return s_a_history

s_a_history = goal_maze_ret_s_a(pi_0)
print(s_a_history)

def update_theta(theta, pi, s_a_history):
  eta = 0.1
  T = len(s_a_history) - 1

  [m,n] = theta.shape
  delta_theta = theta.copy()

 # print(m)
 # print(n)

  for i in range(0, m):
    for j in range(0, n):
      if not( np.isnan(theta[i, j]) ):
        SA_i = [SA for SA in s_a_history if SA[0] == i]
        SA_ij = [SA for SA in s_a_history if SA == [i, j] ]
  #      print('--SA_i--')
  #      print(SA_i)
  #      print('--SA_ij--')
  #      print(SA_ij)
        N_i = len(SA_i)
        N_ij = len(SA_ij)
        delta_theta[i, j] = (N_ij + pi[i, j] * N_i) / T

  new_theta = theta + eta * delta_theta

  return new_theta

new_theta = update_theta(theta_0, pi_0, s_a_history)
print(new_theta)
pi = softmax_convert_into_pi_from_theta(new_theta)
print(pi)

stop_epsilon = 10**-8

theta = theta_0
pi = pi_0
is_continue = True
count = 1

while is_continue:
  s_a_history = goal_maze_ret_s_a(pi)
  new_theta = update_theta(theta, pi, s_a_history)
  new_pi = softmax_convert_into_pi_from_theta(new_theta)

  print(np.sum(np.abs(new_pi - pi)))
  print(len(s_a_history)-1)

  if np.sum(np.abs(new_pi - pi)) < stop_epsilon:
    is_continue = False
  else:
    theta = new_theta
    pi = new_pi

np.set_printoptions(precision=3, suppress=True)
print(pi)

from matplotlib import animation
from IPython.display import HTML

def init():
  line.set_data([],[])
  return (line,)

def animate(i):
  state = s_a_history[i][0]
  x = (state % 3) + 0.5
  y = 2.5 - int(state /3)
  line.set_data(x, y)
  return (line,)

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(s_a_history), interval=200, repeat=False)

HTML(anim.to_jshtml())

[a, b] = theta_0.shape
Q = np.random.rand(a, b) * theta_0

def simple_convert_into_pi_from_theta2(theta):
  [m, n] = theta.shape
  pi = np.zeros((m,n))
  for i in range(0, m):
    pi[i, :] = theta[i, :] / np.nansum(theta[i, :])

  pi = np.nan_to_num(theta[i, :])

  return pi

pi_0 = simple_convert_into_pi_from_theta2(theta_0)
pi_0

def get_action2(s, Q, epsilon, pi_0):
  direction = ["up", "right", "down", "left"]

  if np.random.rand() < epsilon:
    next_direction = np.random.choice(direction, 1, p=pi[s, :])
  else:
    next_direction = direction[np.nanargmax(Q[s, :])]
  #print(next_direction)

  if next_direction == "up":
    actin = 0
  elif next_direction == "right":
    actin = 1
  elif next_direction == "down":
    actin = 2
  elif next_direction == "left":
    actin = 3

  return action

def get_next_s2(s, a, Q, epsilon, pi_0):
  direction = ["up", "right", "down", "left"]

  next_direction = direction[a]
  #print(next_direction)

  if next_direction == "up":
    s_next = s - 3
  elif next_direction == "right":
    s_next = s + 1
  elif next_direction == "down":
    s_next = s + 3
  elif next_direction == "left":
    s_next = s - 1

  return s_next

def Sarsa( s, a, r, s_next, a_next, Q, eta, gamma):
  if s_next == 8:
    Q[s, a] = Q[s, a] + eta * (r - Q[s, a])
  else:
    Q[s, a] = Q[s, a] + eta * ( r + gamma * Q[s_next, a_next] - Q[s, a])

  return Q