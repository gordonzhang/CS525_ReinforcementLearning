from Agent import Agent
import numpy as np


class Environment:

    def __init__(self, width=10, height=10, num_agents=1, start=(1,1), goal=(10,10), view_range=1):
        self.width = width
        self.height = height
        # self.grid stores state of each grid of the map
        # 0-obstacle, 1-walkable, 2-goal, 3-agent
        self.grid = np.ones((height, width), dtype=int)
        self.grid[goal] = 2

        self.num_agents = num_agents

        self.agents = {}
        for n in range(num_agents):
            self.agents[n] = Agent(self)

        self.start = start
        self.goal = goal

        self.view_range = view_range

        self.dead = False

        self.reward_map = np.zeros((height,width))
        self.reward_map[goal] = 1.0

        self.reward_death = -100.0


    def reset(self):
        '''

        :return: dict of agent state
        '''
        self.__init__(self.width, self.height, self.num_agents, self.start, self.goal)

        # FIXME: temp setup states
        states = None
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

        results = {}
        for rid, action in actions.items():
            result = self.make_action(rid, action)
            
            # state_new, reward, done, info = result
            results[rid] = result

        return results

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
            new_pos = (current_pos[0]-1,current_pos[1])
        elif action == 1:
            new_pos = (current_pos[0],current_pos[1]+1)
        elif action == 2:
            new_pos = (current_pos[0]+1,current_pos[1])
        elif action == 3:
            new_pos = (current_pos[0],current_pos[1]-1)

        try:
            assert new_pos >= (0,0), "out of bounds - outside map (below_min)"
            assert new_pos < (self.height, self.width), "out of bounds - outside map (above_max)"
            assert self.grid[new_pos] != 0, "out of bounds - internal edge"

        except Exception as e:
            print("position:", new_pos, "is", e)
            print("\tRemember (in y,x fromat) the grid size is", self.grid.shape)
            self.dead = True
            reward = self.get_reward(new_pos)
            return None, reward, True, None

        state_prime = agent.get_state(new_pos)
        reward = self.get_reward(new_pos)
        done = self.get_terminate(new_pos)

        return state_prime, reward, done, None

    def get_reward(self, pos):
        '''

        :param pos: position of the agent
        :return: dict of tuples, each element being:
            reward: reward for the given position
        '''

        # FIXME: temp implement reward
        
        if(self.dead):
            reward = self.reward_death
        
        reward = self.reward_map[pos]

        return reward

    def get_terminate(self, pos):
        '''

        :param pos: pos of the agent
        :return: is the state an end-state
        '''

        if(pos) == self.goal:
            # at goal
            return True
            
        # TODO: implement agent collision

        return False

    def sense_from_position(self, pos):
        '''

        :param pos: pos of the agent
        :return: sense of the state
        '''

        # action: 0-north, 1-east, 2-south, 3-west
        # 0-obstacle, 1-walkable, 2-goal, 3-agent

        north_lim = max(0,pos[0]-1-self.view_range)
        south_lim = min(self.height,pos[0]+1+self.view_range)
        west_lim = max(0,pos[1]-1-self.view_range)
        east_lim = min(self.width, pos[1]+1+self.view_range)

        north_min = max(0, pos[0])
        south_min = min(self.height, pos[0]+1)
        west_min = max(0, pos[1])
        east_min = min(self.width, pos[1]+1)

        north = self.grid[north_lim:north_min, pos[1]]
        south = self.grid[south_min:south_lim, pos[1]]
        west = self.grid[pos[0], west_lim:west_min]
        east = self.grid[pos[0], east_min:east_lim]

        north = north[::]
        west = west[::]
        
        north = self.sense_helper(north)
        south = self.sense_helper(south)
        west = self.sense_helper(west)
        east = self.sense_helper(east)

        return {0:north, 1:east, 2:south, 3:west}

    def sense_helper(self, arr):
        '''

        :param arr: array of direction view
        :return: array of direction view
        '''

        arr = np.pad(arr, (0,max(0,self.view_range-len(arr))), 'constant', constant_values=(0))

        set_unknown = False

        for i, val in enumerate(arr):
            if set_unknown:
                arr[i] = -1

            elif val == 0 or val == 3:
                set_unknown = True

            elif val == 2:
                arr[i] = 1

        return arr

    def render(self):
        '''

        '''
        pass