#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque, namedtuple
import os

import torch
import torch.nn.functional as f
import torch.optim as optim

from agent import Agent
from dqn_model import DQN
from matplotlib import pyplot as plt


torch.manual_seed(595)
np.random.seed(595)
random.seed(595)


class Agent_DQN(Agent):

    def __init__(self, env, args):
        """
        Initialize everything you need here.
        For example:
            parameters for neural network
            initialize Q net and target Q net
            parameters for replay buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """
        super(Agent_DQN, self).__init__(env)

        if torch.cuda.is_available():
            self.device = 'cuda'
            print("Using GPU!!!!")
        else:
            'cpu'
            print("WARNING")
            print("WARNING")
            print("Using CPU")

        self.state_size = env.get_state()[0].as1xnArray().shape[0]
        self.action_size = 4

        self.memory = deque(maxlen=10000)
        self.thirty_ep_ep = deque(maxlen=10000)
        self.thirty_ep_reward = deque(maxlen=10000)

        # Discount Factor
        self.gamma = 0.99
        # Exploration Rate: at the beginning do 100% exploration
        self.epsilon = 1.0
        # Decay epsilon so we can shift from exploration to exploitation
        self.epsilon_decay = 0.995
        # Set floor for how low epsilon can go
        self.epsilon_min = 0.01
        # Set the learning rate
        self.learning_rate = 0.00015
        # batch_size
        self.batch_size = 32

        self.epsilon_decay_frames = 1.0/500000

        self.policy_net = DQN(self.state_size, self.action_size).to(self.device)
        self.target_net = DQN(self.state_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        self.loss = 0

        self.file_path = 'trained_models_2/./Q_Network_Parameters_'

        with open('trained_models_2/log2.txt', 'w+') as log:
            log.write("episode,avg_reward,epsilon\n")

        if args.test_dqn:
            # load trained model
            print('loading trained model')
            file_number_to_load = 1933
            load_file_path = self.file_path+str(file_number_to_load)+'.pth'
            self.policy_net.load_state_dict(torch.load(load_file_path, map_location=lambda storage, loc: storage))

            # for name, param in self.policy_net.named_parameters():
            # print(name, '\t\t', param.shape)
            print('loaded weights')
            print(self.policy_net.head.weight)


    def train(self, n_episodes=100000):
        ep_epsilon = []
        accumulated_reward = 0
        rewards_30 = []

        for i_episode in range(n_episodes):
            results = self.env.reset()
            state, reward, done, _ = self.unpack(results)
            render = os.path.isfile('.makePicture')


            # Counters for Reward Averages per episode:
            ep_reward = 0.0

            while not done:
                action = self.make_action(state, False)
                results = self.env.step({0: action})
                next_state, reward, done, _ =  self.unpack(results)
                # print(reward, done)
                self.push(state, action, reward, next_state, done)
                state = next_state

                if i_episode > 1000 and len(self.memory) > self.batch_size:
                    self.learn()
                    if i_episode % 5000 == 0:
                        print('------------ UPDATING TARGET -------------')
                        self.target_net.load_state_dict(self.policy_net.state_dict())

                ep_reward += reward
                accumulated_reward += reward

            rewards_30.append(ep_reward)
            # print(rewards_30)
            if len(rewards_30) > 30:
                # print("IN HERE")
                del rewards_30[0]

            ep_epsilon.append(self.epsilon)
            # Print average reward for the episode:
            # print('Episode ', i_episode, 'had a reward of: ', ep_reward)
            # print('Epsilon: ', self.epsilon)

            # Logging the average reward over 30 episodes
            if i_episode % 30 == 0:
                self.thirty_ep_reward.append(accumulated_reward/30.0)
                self.thirty_ep_ep.append(i_episode)
                with open('trained_models_2/log.txt', 'a+') as log:
                    log.write(str(i_episode)+' had a reward of ' + str(accumulated_reward/30.0)+' over 30 ep\n')
                with open('trained_models_2/log2.txt', 'a+') as log:
                    log.write(str(i_episode) + ',' + str(sum(rewards_30)/30.0) + ',' + str(self.epsilon) + '\n')

                accumulated_reward = 0.0
                # Save network weights after we have started to learn
                if i_episode > 3000 and i_episode % 2000 == 0:

                    print('saving... ', i_episode)
                    save_file_path = self.file_path+str(i_episode)+'.pth'
                    torch.save(self.policy_net.state_dict(), save_file_path)


                fig = plt.figure()
                plt.plot(ep_epsilon)
                plt.title('Epsilon decay')
                plt.xlabel('Episodes')
                plt.ylabel('Epsilon Value')
                plt.savefig('trained_models_2/epsilon.png')
                plt.close()

                fig = plt.figure()
                plt.plot(self.thirty_ep_ep, self.thirty_ep_reward)
                plt.title('Average Reward per 30 Episodes')
                plt.xlabel('Episodes')
                plt.ylabel('Average Reward')
                plt.savefig('trained_models_2/reward.png')
                plt.close()

            if i_episode % 200 == 0:
                print('Episode: ',i_episode ,'Avg reward of last 30 episodes: ', sum(rewards_30)/30.0)

    def learn(self):
        sampled_batch = self.replay_buffer(self.batch_size)

        states, actions, rewards, next_states, dones = list(zip(*sampled_batch))

        states = torch.from_numpy(np.stack(states)).to(self.device)
        actions = torch.from_numpy(np.stack(actions)).to(self.device)
        rewards = torch.from_numpy(np.stack(rewards)).to(self.device)
        next_states = torch.from_numpy(np.stack(next_states)).to(self.device)
        dones = torch.from_numpy(np.stack(dones)).to(self.device)
        
        states = states.float()
        next_states = next_states.float()
        actions = actions.unsqueeze(1)
        qfun = self.policy_net(states)

        state_action_values = qfun.gather(1, actions.long()).squeeze()

        next_state_values = self.target_net(next_states).max(1).values.detach()

        TD_error = rewards + self.gamma*next_state_values*(1-dones)

        self.loss = f.smooth_l1_loss(state_action_values, TD_error)

        self.optimizer.zero_grad()
        self.loss.backward()

        # for param in self.policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        observation = torch.tensor(observation, dtype=torch.float32).to(self.device)
        observation = observation.unsqueeze(0)

        if not test:
            if np.random.rand() <= self.epsilon:
                action = random.randrange(self.action_size)
            else:
                with torch.no_grad():
                    # action = torch.argmax(self.policy_net(observation)).item()
                    action = self.target_net(observation).max(1)[1].view(1, 1).item()
                    # print(action)

            if self.epsilon > self.epsilon_min:
                self.epsilon = max(0, self.epsilon - self.epsilon_decay_frames)
        else:
            with torch.no_grad():
                action = torch.argmax(self.policy_net(observation)).item()
        return action

    def push(self, state, action, reward, next_state, done):
        """
        Push new data to buffer and remove the old one if the buffer is full.
        """
        action = np.array(action, dtype=np.uint8)
        reward = np.array(reward, dtype=np.float32)
        done = np.array(done, dtype=np.float32)
        self.memory.append((state, action, reward, next_state, done))

    def replay_buffer(self, batch_size):
        """
        Select batch from buffer.
        """
        return random.sample(self.memory, batch_size)

    def unpack(self, results):
        result = results[0]
        state, reward, done, info = result.asTuple()
        return state.as1xnArray(), reward, done, info

    def init_game_setting(self):
        pass
