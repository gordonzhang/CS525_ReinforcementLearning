from Environment import Environment
import time

env = Environment(width=5, height=5, num_agents=1, start=(0,0), goal=(4,4), view_range=2, render=True, std=True)
time.sleep(1)

# action_list = [2,0,1,3]
action_list = [2,2,2,2,1,1,1,1,0]

for a in action_list:
    agents = env.step({0:a})
    state, reward, done, _ = agents[0]
    time.sleep(1)
    
    if done:
        break