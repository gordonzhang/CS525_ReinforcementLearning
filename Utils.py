import numpy as np
from enum import Enum
from queue import Queue


class Position:
    def __init__(self, y=0, x=0):
        self.x = x
        self.y = y

    def up(self):
        return Position(self.y-1, self.x)
    def down(self):
        return Position(self.y+1, self.x)
    def right(self):
        return Position(self.y, self.x+1)
    def left(self):
        return Position(self.y, self.x-1)

    def asTuple(self):
        return self.y, self.x

    def __lt__(self, o):  # For x < y
        return self.x < o.x and self.y < o.y
    def __le__(self, o):  # For x <= y
        return self.x <= o.x and self.y <= o.y
    def __eq__(self, o):  # For x == y
        return self.x == o.x and self.y == o.y
    def __ne__(self, o):  # For x != y OR x <> y
        return self.x != o.x and self.y != o.y
    def __gt__(self, o):  # For x > y
        return self.x > o.x and self.y > o.y
    def __ge__(self, o):  # For x >= y
        return self.x >= o.x and self.y >= o.y

    def __str__(self):
        return "(Y: %s, X: %s)" % (self.y, self.x)

    def __repr__(self):
        return str(self)


class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class Observation:
    def __init__(self, up, right, down, left):
        self.up = up
        self.right = right
        self.down = down
        self.left = left

    def asDict(self):
        return {
            Direction.UP: self.up,
            Direction.RIGHT: self.right,
            Direction.DOWN: self.down,
            Direction.LEFT: self.left
        }

    def asStd(self):
        return {
            0: self.up.tolist(),
            1: self.right.tolist(),
            2: self.down.tolist(),
            3: self.left.tolist()
        }


class State:
    def __init__(self, observation: Observation, pastPositions):
        self.observation = observation
        self.pastPositions = pastPositions

    def asStd(self):
        # obs = {}
        # for o in self.observations.keys():
        #     obs[o.value] = self.observations[o]
        obs = self.observation.asStd()

        pp = []
        for p in self.pastPositions:
            pp.append(p.asTuple())

        return obs, pp

    def as1xnArray(self):
        obs = self.observation
        
        arr = []
        arr.extend(obs.up)
        arr.extend(obs.right)
        arr.extend(obs.down)
        arr.extend(obs.left)

        for pos in self.pastPositions:
            arr.append(pos.x)
            arr.append(pos.y)

        return np.array(arr)

    def as2xnArray(self):
        assert len(self.observation.up) == 2, "observation len is not 2, so cannot make 2xn array"

        obs = self.observation
        
        arr = [obs.up, obs.right, obs.down, obs.left]

        for pos in self.pastPositions:
            arr.append(np.array([pos.y, pos.x]))
            
        return np.array(arr)


class RewardMap:
    def __init__(self, goal, height, width, horizon, goal_reward):
        self.horizon = horizon
        self.goal = goal
        self.width = width
        self.height = height

        self.goal_reward = goal_reward

        self.reward_map = np.zeros((height, width))

        self._populate_map()

    def _reward(self, pos):
        return self.reward_map[pos.asTuple()]

    def _populate_map(self):
        # get max manhattan distance
        dist_max = -1

        for y in range(self.height):
            for x in range(self.width):
                dist_max = max(dist_max, self._manhattan_distance(Position(y, x), self.goal))

        divisor = dist_max / self.horizon

        for y in range(self.height):
            for x in range(self.width):
                numerator = self._manhattan_distance(Position(y, x), self.goal)
                self.reward_map[y, x] = -numerator//self.horizon/divisor

        self.reward_map[self.goal.asTuple()] = self.goal_reward

        # print(self.reward_map)

    def _manhattan_distance(self, pos0, pos1):
        # dist = |𝑎−𝑐|+|𝑏−𝑑|
        return abs(pos0.x - pos1.x) + abs(pos0.y - pos1.y)


class StepResponse:
    def __init__(self, state, reward, done, info=None):
        self.state = state
        self.reward = reward
        self.done = done
        self.info = info
    
    def asTuple(self):
        return self.state, self.reward, self.done, self.info
    
    def asStd(self):
        if self.state is None:
            return None, self.reward, self.done, self.info
            
        return self.state.asStd(), self.reward, self.done, self.info


class Agent:
    def __init__(self, env, pos_hist_len=5, pos_start=Position()):
        self.pos = pos_start
        self.pos_history = Queue(maxsize=pos_hist_len)

        self.env = env

        for i in range(pos_hist_len):
            self.pos_history.put(pos_start, block=False)

    def _set_pos(self, new_pos):
        self.pos = new_pos
        
    def _get_state(self, new_pos):
        """

        :param new_pos: new position of the agent
        :return: tuple of 2 items:
            obs: sensed observation from the environment
            position_history: history of visited positions
        """
        self.pos_history.get(block=False)
        self.pos_history.put(new_pos, block=False)

        observation = self.env._sense_from_position(new_pos)

        state = State(observation, list(self.pos_history.queue))

        return state
