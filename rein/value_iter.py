from collections import defaultdict
from gridworld import GridWorld
from policy_iter import greedy_policy

def value_iter_onestep(V, env, gamma):
    print("---value iter onestep---")
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0
            continue

        action_values=[]
        for action in env.actions():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            value = r + gamma*V[next_state]
            action_values.append(value)

        V[state] = max(action_values)
        #print(V[state])

    #print(f'--- {V}')
    return V

def value_iter(V, env, gamma, threshold=0.001, is_render=True):
    print('---value iter---')

if __name__ == "__main__":
    print("----start-----")
    V = defaultdict(lambda: 0)
    env = GridWorld()
    gamma = 0.9

    V = value_iter_onestep(V, env, gamma)
    print(V)
    
    V = value_iter(V, env, gamma)

#    pi = greedy_policy(V, env, gamma)

    #env.render_v(V, pi)

    print("----end-----")    
    
