#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import numpy as np
from Environment import Environment

seed = 11037


def parse():
    parser = argparse.ArgumentParser(description="DS595/CS525 RL Project 4")
    parser.add_argument('--test_dqn', action='store_true', help='whether test DQN')

    return parser.parse_args()


def test(agent, env, total_episodes=30):
    rewards = []
    env.seed(seed)
    for i in range(total_episodes):
        state = env.reset()
        agent.init_game_setting()
        done = False
        episode_reward = 0.0

        # playing one game
        while not done:
            action = agent.make_action(state, test=True)
            state, reward, done, info = env.step(action)
            episode_reward += reward

        rewards.append(episode_reward)
    print('Run %d episodes' % total_episodes)
    print('Mean:', np.mean(rewards))


def run(args):
    if args.test_dqn:
        env = Environment('BreakoutNoFrameskip-v4', args, atari_wrapper=True, test=True)
        from agent_dqn import Agent_DQN
        agent = Agent_DQN(env, args)
        test(agent, env, total_episodes=100)


if __name__ == '__main__':
    arguments = parse()
    run(arguments)