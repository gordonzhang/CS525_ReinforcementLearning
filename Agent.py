import numpy as np

class Agent:

    def __init__(self, env, pos_start=(0, 0)):
        self.pos = pos_start
        self.pos_history = [pos_start]
        self.env = env

    def set_pos(self, new_pos):
        self.pos = new_pos
        
    def get_state(self, new_pos):
        '''

        :param new_pos: new position of the agent
        :return: tuple of 2 items:
            obs: sensed observation from the environment
            position_history: history of visited positions
        '''
        self.pos_history.append(new_pos)

        obs = self.env.sense_from_position(new_pos)
        
        return obs, self.pos_history