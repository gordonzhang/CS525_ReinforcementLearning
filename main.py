import argparse
from test import test
from Environment import Environment

from Utils import Position

def parse():
    parser = argparse.ArgumentParser(description="DS595/CS525 RL Project 3")
    parser.add_argument('--env_name', default=None, help='environment name')
    parser.add_argument('--train_dqn', action='store_true', help='whether train DQN')
    parser.add_argument('--test_dqn', action='store_true', help='whether test DQN')
    # parser.add_argument('--batch_size', type=int, help='batch size')

    args = parser.parse_args()
    return args


def run(args):
    if args.train_dqn:
        env = Environment(width=5, height=5, num_agents=1, start=Position(0, 0), goal=Position(4, 4), view_range=2)
        from agent_dqn import Agent_DQN
        agent = Agent_DQN(env, args)
        agent.train()

    if args.test_dqn:
        env = Environment()
        from agent_dqn import Agent_DQN
        agent = Agent_DQN(env, args)
        test(agent, env, total_episodes=100)


if __name__ == '__main__':
    arguments = parse()
    run(arguments)
