import numpy as np

class Agent:

    def __init__(self, pos_start=(0, 0)):
        self.pos = pos_start
        self.pos_history = []
        self.pos_history.append(pos_start)