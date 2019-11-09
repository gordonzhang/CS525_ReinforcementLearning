from Agent import Agent
import numpy as np


class Environment:

    def __init__(self, width=10, height=10, num_agents=1):
        self.width = width
        self.height = height
        # self.grid stores state of each grid of the map
        # 0-obstacle, 1-walkable, 2-goal
        self.grid = np.ones((width, height), dtype=int)
        self.grid[2, 2] = 0
        self.grid[2, 3] = 0

        self.num_agents = num_agents

        self.agents = {}
        for n in range(num_agents):
            self.agents[n] = Agent()


    def reset(self):
        '''

        :return: dict of agent state
        '''
        return states


    def step(self, actions):
        '''

        :param actions: dict of actions per agent
        :return: tuple of following:
            states_prime: dict of new state for each agent
            rewards: dict of reward per agent
            done: dict of agent terminated or not
            info: anything else
        '''

        for agent_id, action in actions.items():
            state_new, reward, done, info = self.make_action(agent_id, action)

        return 0, 0, 0, 0

    def make_action(self, agent_id, action):
        current_pos = self.agents[agent_id].pos
        # action: 0-north, 1-east, 2-south, 3-west

        new_pos = (-1,-1)
        if action == 0:
            new_pos = (current_pos[0],current_pos[1]-1)
        elif action == 1:
            new_pos = (current_pos[0]+1,current_pos[1])
        elif action == 2:
            new_pos = (current_pos[0],current_pos[1]+1)
        elif action == 3:
            new_pos = (current_pos[0]-1,current_pos[1])

        try:
            print(new_pos)
            print(self.grid[new_pos])
            assert new_pos < (0,0), "out of bounds - outside map"
            assert new_pos >= (self.width, self.height), "out of bounds - outside map"
            assert self.grid[new_pos] != 0, "out of bounds - internal edge"
        except Exception as e:
            print(e)
            return "terminate"

        return 0,0,0,0


    def get_range_from_position(self):
        return state


if __name__ == '__main__':
    env = Environment()
    print(env.grid)
    env.agents[0].pos = (2,9)
    env.step({0:1})
