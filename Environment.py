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
            self.agents[n] = Agent(self)


    def reset(self):
        '''

        :return: dict of agent state
        '''
        return states


    def step(self, actions):
        '''

        :param actions: dict of actions per agent
        :return: dict of tuples, each element being:
            state_prime: new state for the agent
            reward: reward for the agent
            done: agent terminated or not
            info: anything else
        '''

        for rid, action in actions.items():
            state_new, reward, done, info = self.make_action(rid, action)

        return 0, 0, 0, 0

    def make_action(self, rid, action):
        '''

        :param rid: id of the agent
        :param action: action for the agent
        :return: tuple of following:
            state_prime: new state for the agent
            reward: reward for the agent
            done: agent terminated or not
            info: anything else
        '''
        agent = self.agents[rid]
        current_pos = agent.pos
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
            assert new_pos < (0,0), "out of bounds - outside map"
            assert new_pos >= (self.width, self.height), "out of bounds - outside map"
            assert self.grid[new_pos] != 0, "out of bounds - internal edge"

        except Exception as e:
            print("position:", new_pos, "is", e)
            reward = self.get_reward(new_pos)
            return None, reward, True, None

        state_prime = agent.get_state(new_pos)
        reward = self.get_reward(new_pos)
        done = self.get_terminate(new_pos)

        return state_prime, reward, done, None

    def get_reward(self, new_pos):
        '''

        :param new_pos: new position of the agent
        :return: dict of tuples, each element being:
            reward: reward for the given position
        '''

        # FIXME: temp set reward
        reward = 0

        return reward

    def sense_from_position(self, rid):
        '''

        :param rid: id of the agent
        :return: sense of the state
        '''

        # FIXME: temp set state
        state = {0:1, 1:1, 2:1, 3:1}

        return state


if __name__ == '__main__':
    env = Environment()
    print(env.grid)
    env.agents[0].pos = (2,9)
    env.step({0:1})
