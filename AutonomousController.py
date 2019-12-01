from Environment import Environment
import time
from Utils import Position


env = Environment(width=5, height=5, num_agents=1, start=Position(1, 1), goal=Position(4, 4), view_range=3, render=True)
print(env.agents[0].pos_history.queue)
agents = env.reset()
agent = agents[0]
state, reward, done, _ = agent.state, agent.reward, agent.done, agent.info
print(state.as1xnArray())
time.sleep(1)

# action_list = [2,0,1,3]
action_list = [2, 2, 2, 2, 1, 1, 1, 1, 0]

for a in action_list:
    agents = env.step({0: a})
    agent = agents[0]
    state, reward, done, _ = agent.state, agent.reward, agent.done, agent.info
    # print(state, reward, done)
    time.sleep(1)
    
    if done:
        break
