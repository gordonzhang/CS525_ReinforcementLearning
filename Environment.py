from Utils import Agent, Direction, Position, StepResponse, Observation, RewardMap
from enum import Enum
import numpy as np


class Environment:

    def __init__(self, height=5, width=5, num_agents=1, start=Position(0, 0), goal=Position(9E9, 9E9),
                 view_range=1, horizon=2, goal_reward=1.0, pos_hist_len=5, render=False, std=False):
        self.std = std

        self.width = width
        self.height = height

        self.pos_hist_len = pos_hist_len

        if self.std:
            start = Position(y=start[0], x=start[1])
            goal = Position(y=goal[0], x=goal[1])

        if goal.x > self.width or goal.y > self.height:
            goal = Position(self.height-1, self.width-1)

        # self.grid stores state of each grid of the map
        # 0-obstacle, 1-walkable, 2-goal, 3-agent
        self.grid = np.ones((height, width), dtype=int)
        self.grid[goal.asTuple()] = 2

        self.num_agents = num_agents

        self.agents = {}
        for n in range(num_agents):
            self.agents[n] = Agent(self, pos_hist_len=self.pos_hist_len)

        self.start = start
        self.goal = goal

        self.view_range = view_range

        self.dead = False

        self.reward_map = RewardMap(self.goal, self.height, self.width, horizon, goal_reward)

        # self.reward_map = np.zeros((height,width))
        # self.reward_map[goal.asTuple()] = 1.0

        self.reward_death = -100.0

        self.renderEnv = render
        
        self.called_render = False

        if self.renderEnv:
            self._render()

    def get_state(self):        
        states = {}
        print(self.agents)
        for rid, agent in self.agents.items():
            print(agent)
            state = agent._get_state(agent.pos)

            if self.std:
                state = state.asStd()

            states[rid] = state

        if self.renderEnv:
            self._render()

        return states


    def reset(self):
        """

        :return: dict of agent state
        """
        self.__init__(self.width, self.height, self.num_agents, self.start, self.goal)

        # FIXME: temp setup states
        states = None
        return states

    def step(self, actions):
        """

        :param actions: dict of actions per agent
        :return: dict of tuples, each element being:
            state_prime: new state for the agent
            reward: reward for the agent
            done: agent terminated or not
            info: anything else
        """

        results = {}
        for rid, action in actions.items():
            if self.std and isinstance(action, Enum):
                action = action.value
            result = self._make_action(rid, action)

            if self.std:
                result = result.asStd()
            
            # state_new, reward, done, info = result
            results[rid] = result

        if self.renderEnv:
            self._render()

        return results

    def _make_action(self, rid, action):
        """

        :param rid: id of the agent
        :param action: action for the agent
        :return: tuple of following:
            state_prime: new state for the agent
            reward: reward for the agent
            done: agent terminated or not
            info: anything else
        """
        agent = self.agents[rid]
        pos = agent.pos
        # action: 0-north, 1-east, 2-south, 3-west

        new_pos = Position(-1, -1)
        if action == Direction.UP or action == Direction.UP.value:
            new_pos = pos.up()
        elif action == Direction.RIGHT or action == Direction.RIGHT.value:
            new_pos = pos.right()
        elif action == Direction.DOWN or action == Direction.DOWN.value:
            new_pos = pos.down()
        elif action == Direction.LEFT or action == Direction.LEFT.value:
            new_pos = pos.left()

        agent._set_pos(new_pos)

        try:
            assert new_pos >= Position(0, 0), "out of bounds - outside map (below_min)"
            assert new_pos < Position(self.height, self.width), "out of bounds - outside map (above_max)"
            assert self.grid[new_pos.asTuple()] != 0, "out of bounds - internal edge"

        except Exception as e:
            # print("position:", new_pos, "is", e)
            # print("\tRemember (in y,x format) the grid size is", self.grid.shape)
            self.dead = True
            reward = self._get_reward(new_pos)
            return StepResponse(None, reward, True)
            # return None, reward, True, None

        state_prime = agent._get_state(new_pos)
        reward = self._get_reward(new_pos)
        done = self._get_terminate(new_pos)

        resp = StepResponse(state_prime, reward, done)
        
        return resp

    def _get_reward(self, pos):
        """

        :param pos: position of the agent
        :return: dict of tuples, each element being:
            reward: reward for the given position
        """

        # FIXME: temp implement reward

        if self.dead:
            reward = self.reward_death
            return reward
        
        # if pos[0] < 0 or pos[1] < 0 or pos[0] > self.height-1 or pos[1] < :
        reward = self.reward_map._reward(pos)

        return reward

    def _get_terminate(self, pos):
        """

        :param pos: pos of the agent
        :return: is the state an end-state
        """

        if pos == self.goal:
            # at goal
            return True
            
        # TODO: implement agent collision

        return False

    def _sense_from_position(self, pos):
        """

        :param pos: pos of the agent
        :return: sense of the state
        """

        # action: 0-north, 1-east, 2-south, 3-west
        # 0-obstacle, 1-walkable, 2-goal, 3-agent

        north_lim = max(0, pos.up().y - self.view_range)
        south_lim = min(self.height, pos.down().y + self.view_range)
        west_lim = max(0, pos.left().x - self.view_range)
        east_lim = min(self.width, pos.right().x + self.view_range)

        north_min = max(0, pos.y)
        south_min = min(self.height, pos.down().y)
        west_min = max(0, pos.x)
        east_min = min(self.width, pos.right().x)

        north = self.grid[north_lim:north_min, pos.x]
        south = self.grid[south_min:south_lim, pos.x]
        west = self.grid[pos.y, west_lim:west_min]
        east = self.grid[pos.y, east_min:east_lim]

        north = north[::]
        west = west[::]
        
        north = self._sense_helper(north)
        south = self._sense_helper(south)
        west = self._sense_helper(west)
        east = self._sense_helper(east)

        obs = Observation(north, east, south, west)

        return obs

        # return {Direction.UP:north, Direction.RIGHT:east, Direction.DOWN:south, Direction.LEFT:west}

    def _sense_helper(self, arr):
        """

        :param arr: array of direction view
        :return: array of direction view
        """

        arr = np.pad(arr, (0, max(0, self.view_range-len(arr))), 'constant', constant_values=0)

        set_unknown = False

        for i, val in enumerate(arr):
            if set_unknown:
                arr[i] = -1

            elif val == 0 or val == 3:
                set_unknown = True

            elif val == 2:
                arr[i] = 1

        if len(arr) > self.view_range:
            arr = arr[:self.view_range]

        return arr

    def _render(self, block=False):
        """

        """


        import matplotlib.pyplot as plt
        from matplotlib import colors
        import matplotlib.ticker as plticker
        from matplotlib.ticker import AutoMinorLocator

        if not self.called_render:
            self.called_render = True

            # persistent graph variables
            self.fig, self.ax = plt.subplots()

            self.cmap = colors.ListedColormap(['black', 'white', 'green', 'blue'])
            self.bounds = [0, 0, 1, 1, 2, 2, 3, 3]
            self.norm = colors.BoundaryNorm(self.bounds, self.cmap.N)
            
            temp = self.grid.copy()
            vpad = np.zeros((self.height, 1))
            hpad = np.zeros((1, self.width+2))

            self.grid_copy = np.hstack((vpad, temp, vpad))
            self.grid_copy = np.vstack((hpad, self.grid_copy, hpad))

            self.ax.set_xlim(0, self.width+1.5)
            self.ax.set_ylim(self.height+1.5, 0)
            self.ax.set_xticks(range(0, self.width+2))
            x_values = np.arange(start=0, stop=self.width+2) - .5
            self.ax.set_xticks(x_values, minor=True)

            self.ax.set_yticks(range(self.height+1, -1, -1))
            y_values = np.arange(start=self.height+2, stop=-1, step=-1) - .5
            self.ax.set_yticks(y_values, minor=True)
            self.ax.grid(which='minor')

            # self.ax.invert_yaxis()

            self.im = self.ax.imshow(self.grid_copy, cmap=self.cmap, norm=self.norm)

        self.grid_copy[1:self.height+1, 1:self.width+1] = self.grid

        for rid, agent in self.agents.items():
            pos = agent.pos
            pos = pos.down().right()
            self.grid_copy[pos.asTuple()] = 3

        # grid_copy = self.grid_copy[::-1,:]
        # self.im.set_data(grid_copy)
        self.im.set_data(self.grid_copy)

        self.fig.suptitle("Agent in grid world, plus edges")
        plt.draw()
        plt.show(block=block)
        # if (iteration % 2 == 0 and iteration <= 8) or iteration == 40:
        # 	fig.savefig("P=%.1f_t=%d_Iteration=%d.png" % (P, t, iteration))
        plt.pause(0.001)


if __name__ == "__main__":
    env = Environment(width=20, height=20, num_agents=1, start=Position(0, 0), view_range=2, render=True)

    print(env.get_state())
    
    agents = env.step({0: Direction.RIGHT})
    stepResponse = agents[0]
    state = stepResponse.state
    print(state.pastPositions)
    print("---1")

    res = state.as2xnArray()
    print(res)

    env._render(block=True)
