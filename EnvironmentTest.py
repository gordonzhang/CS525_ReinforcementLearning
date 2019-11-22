import unittest
from Environment import Environment
from Utils import Agent, Direction, Position, StepResponse, Observation
import numpy as np

class TestEnvironment(unittest.TestCase):

    def setUp(self):
        self.env_2x1 = Environment(width=2, height=1, num_agents=1, start=Position(0,0), goal=Position(0,1), view_range=1)
        self.env_5x5 = Environment(width=5, height=5, num_agents=1, start=Position(0,0), goal=Position(4,4), view_range=2)


        self.env_3x3_std = Environment(width=3, height=3, num_agents=1, start=(0,0), goal=(2,2), view_range=1, std=True)

    def test_2x1_init_and_reset(self):
        grid_expected = np.array( [[1,2]] )
        
        grid_actual = self.env_2x1.grid
        self.assertEqual(grid_actual.all(), grid_expected.all())
        
        self.env_2x1.reset()
        grid_actual = self.env_2x1.grid
        self.assertEqual(grid_actual.all(), grid_expected.all())

    def test_2x1_sense(self):
        state_actual = self.env_2x1._sense_from_position(Position(0,0))
        state_actual = state_actual.asStd()

        state_expected = {0:[0], 1:[1], 2:[0], 3:[0]}
        self.assertEqual(state_actual, state_expected)

        state_actual = self.env_2x1._sense_from_position(Position(0,1))
        state_actual = state_actual.asStd()

        state_expected = {0:[0], 1:[0], 2:[0], 3:[1]}
        self.assertEqual(state_actual, state_expected)
        
    def test_2x1_step(self):
        # action: 0-north, 1-east, 2-south, 3-west
        agents = self.env_2x1.step({0:Direction.RIGHT}) # agent 0 : go east
        resp = agents[0]
        s_prime_actual, reward_actual, done_actual, _ = resp.asStd()

        sense_actual = s_prime_actual[0]
        sense_expected = {0:[0], 1:[0], 2:[0], 3:[1]}
        self.assertEqual(sense_actual, sense_expected)

        history_actual = s_prime_actual[1]
        history_expected = [(0,0),(0,1)]
        self.assertEqual(history_actual, history_expected)

        reward_expected = 1.0
        self.assertEqual(reward_actual, reward_expected)

        done_expected = True
        self.assertEqual(done_actual, done_expected)

    def test_5x5_sense(self):
        state_actual = self.env_5x5._sense_from_position(Position(0,0))
        state_actual = state_actual.asStd()
        
        state_expected = {0:[0,-1], 1:[1,1], 2:[1,1], 3:[0,-1]}

        self.assertEqual(state_actual, state_expected)

        state_actual = self.env_5x5._sense_from_position(Position(0,1))
        state_actual = state_actual.asStd()
        
        state_expected = {0:[0,-1], 1:[1,1], 2:[1,1], 3:[1,0]}
        self.assertEqual(state_actual, state_expected)

        state_actual = self.env_5x5._sense_from_position(Position(0,4))
        state_actual = state_actual.asStd()
        
        state_expected = {0:[0,-1], 1:[0,-1], 2:[1,1], 3:[1,1]}
        self.assertEqual(state_actual, state_expected)

        state_actual = self.env_5x5._sense_from_position(Position(4,4))
        state_actual = state_actual.asStd()
        
        state_expected = {0:[1,1], 1:[0,-1], 2:[0,-1], 3:[1,1]}
        self.assertEqual(state_actual, state_expected)

    def test_env_3x3_std(self):
        # go right
        res = self.env_3x3_std.step({0:1})
        res = res[0]

        self.assertEqual(len(res), 4)

        state, reward, done, info = res
        self.assertEqual(state, ({0:[0], 1:[1], 2:[1], 3:[1]}, [(0,0),(0,1)]))
        self.assertEqual(reward, 0.0)
        self.assertEqual(done, False)
        self.assertEqual(info, None)

        # go right
        res = self.env_3x3_std.step({0:1})
        state, reward, done, info = res[0]
        print(state)

        self.assertEqual(state, ({0:[0], 1:[0], 2:[1], 3:[1]}, [(0,0),(0,1),(0,2)]))
        self.assertEqual(reward, 0.0)
        self.assertEqual(done, False)
        self.assertEqual(info, None)

        # go right
        res = self.env_3x3_std.step({0:1})
        state, reward, done, info = res[0]

        self.assertEqual(state, None)
        self.assertEqual(reward, -100.0)
        self.assertEqual(done, True)
        self.assertEqual(info, None)

if __name__ == '__main__':
    unittest.main()