import unittest
from Environment import Environment
import numpy as np

class TestEnvironment(unittest.TestCase):

    def setUp(self):
        self.env_2x1 = Environment(width=2, height=1, num_agents=1, start=(0,0), goal=(0,1), view_range=1)
        self.env_5x5 = Environment(width=5, height=5, num_agents=1, start=(0,0), goal=(4,4), view_range=2)

    def test_2x1_init_and_reset(self):
        grid_expected = np.array( [[1,2]] )
        
        grid_actual = self.env_2x1.grid
        self.assertEqual(grid_actual.all(), grid_expected.all())
        
        self.env_2x1.reset()
        grid_actual = self.env_2x1.grid
        self.assertEqual(grid_actual.all(), grid_expected.all())

    def test_2x1_sense(self):
        state_actual = self.env_2x1.sense_from_position((0,0))
        state_expected = {0:[0], 1:[1], 2:[0], 3:[0]}
        self.assertEqual(state_actual, state_expected)

        state_actual = self.env_2x1.sense_from_position((0,1))
        state_expected = {0:[0], 1:[0], 2:[0], 3:[1]}
        self.assertEqual(state_actual, state_expected)
        
    def test_2x1_step(self):
        # action: 0-north, 1-east, 2-south, 3-west
        agents = self.env_2x1.step({0:1}) # agent 0 : go east
        s_prime_actual, reward_actual, done_actual, _ = agents[0]

        sense_actual = s_prime_actual[0]
        sense_expected = {0:[0], 1:[0], 2:[0], 3:[1]}
        self.assertEqual(sense_actual, sense_expected)

        history_actual = s_prime_actual[1]
        history_expected = [(0,0),(0,1)]
        self.assertEqual(history_actual, history_expected)

        # TODO: Implement reward
        reward_expected = 0
        self.assertEqual(reward_actual, reward_expected)

        done_expected = True
        self.assertEqual(done_actual, done_expected)

    def test_5x5_sense(self):
        state_actual = self.env_5x5.sense_from_position((0,0))
        state_expected = {0:np.array([0,-1]), 1:np.array([1,1]), 2:np.array([1,1]), 3:np.array([0,-1])}

        for i in range(4):
            self.assertEqual(state_actual[i].all(), state_expected[i].all())

        state_actual = self.env_5x5.sense_from_position((0,1))
        state_expected = {0:np.array([0,-1]), 1:np.array([1,1]), 2:np.array([1,1]), 3:np.array([1,0])}
        for i in range(4):
            self.assertEqual(state_actual[i].all(), state_expected[i].all(), "i: %i, a: %s, e: %s" %(i,state_actual[i], state_expected[i]))

        state_actual = self.env_5x5.sense_from_position((0,4))
        state_expected = {0:np.array([0,-1]), 1:np.array([0,-1]), 2:np.array([1,1]), 3:np.array([1,1])}
        for i in range(4):
            self.assertEqual(state_actual[i].all(), state_expected[i].all(), "i: %i, a: %s, e: %s" %(i,state_actual[i], state_expected[i]))

        state_actual = self.env_5x5.sense_from_position((4,4))
        state_expected = {0:np.array([1,1]), 1:np.array([0,-1]), 2:np.array([0,-1]), 3:np.array([1,1])}
        for i in range(4):
            self.assertEqual(state_actual[i].all(), state_expected[i].all(), "i: %i, a: %s, e: %s" %(i,state_actual[i], state_expected[i]))

if __name__ == '__main__':
    unittest.main()