import numpy as np
from collections import deque
import time
import torch

class Logger(object):
    def __init__(self, n_envs, obs_shape, obs_full_shape, recurrent_hidden_state_size, device='cpu'):
        self.start_time = time.time()
        self.n_envs = n_envs

        self.obs = {}
        self.obs_sum = {}
        self.obs0 = {}
        self.env_steps = {}
        self.eval_recurrent_hidden_states = {}
        self.eval_recurrent_hidden_states1 = {}
        self.eval_recurrent_hidden_states2 = {}
        self.eval_recurrent_hidden_states_maxEnt = {}
        self.eval_masks = {}
        self.last_action = {}

        self.obs_vec = {}
        self.obs_vec['train_eval'] = []
        for i in range(n_envs):
            self.obs_vec['train_eval'].append([])

        self.obs_vec['test_eval'] = []
        for i in range(n_envs):
            self.obs_vec['test_eval'].append([])


        self.obs['train_eval'] = torch.zeros(self.n_envs, *obs_shape)
        self.obs_sum['train_eval'] = torch.zeros(self.n_envs, *obs_full_shape, device=device)
        self.obs0['train_eval'] = torch.zeros(self.n_envs, *obs_full_shape)
        self.env_steps['train_eval'] = torch.ones(self.n_envs, 1)
        self.eval_recurrent_hidden_states['train_eval'] = torch.zeros(self.n_envs, recurrent_hidden_state_size, device=device)
        self.eval_recurrent_hidden_states1['train_eval'] = torch.zeros(self.n_envs, recurrent_hidden_state_size, device=device)
        self.eval_recurrent_hidden_states2['train_eval'] = torch.zeros(self.n_envs, recurrent_hidden_state_size, device=device)
        self.eval_recurrent_hidden_states_maxEnt['train_eval'] = torch.zeros(self.n_envs, recurrent_hidden_state_size, device=device)
        self.eval_masks['train_eval'] = torch.ones(self.n_envs, 1, device=device)
        self.last_action['train_eval'] = torch.full([n_envs, 1], 7, device=device)

        self.obs['test_eval'] = torch.zeros(self.n_envs, *obs_shape)
        self.obs_sum['test_eval'] = torch.zeros(self.n_envs, *obs_full_shape, device=device)
        self.obs0['test_eval'] = torch.zeros(self.n_envs, *obs_full_shape)
        self.env_steps['test_eval'] = torch.ones(self.n_envs, 1)
        self.eval_recurrent_hidden_states['test_eval'] = torch.zeros(self.n_envs, recurrent_hidden_state_size, device=device)
        self.eval_recurrent_hidden_states1['test_eval'] = torch.zeros(self.n_envs, recurrent_hidden_state_size, device=device)
        self.eval_recurrent_hidden_states2['test_eval'] = torch.zeros(self.n_envs, recurrent_hidden_state_size, device=device)
        self.eval_recurrent_hidden_states_maxEnt['test_eval'] = torch.zeros(self.n_envs, recurrent_hidden_state_size, device=device)
        self.eval_masks['test_eval'] = torch.ones(self.n_envs, 1, device=device)
        self.last_action['test_eval'] = torch.full([n_envs, 1], 7, device=device)

        self.episode_rewards = []
        self.episode_rewards_val = []
        self.episode_rewards_test = []
        self.episode_rewards_train = []
        for _ in range(n_envs):
            self.episode_rewards.append([])
            self.episode_rewards_train.append([])
            self.episode_rewards_test.append([])

        self.episode_len_buffer = deque(maxlen = n_envs)
        self.episode_len_buffer_train = deque(maxlen=n_envs)
        self.episode_len_buffer_test = deque(maxlen=n_envs)
        self.episode_reward_buffer = deque(maxlen = n_envs)
        self.episode_reward_buffer_train = deque(maxlen=n_envs)
        self.episode_reward_buffer_test = deque(maxlen=n_envs)

        self.num_episodes = 0
        self.num_episodes_train = 0
        self.num_episodes_test = 0

    def feed_eval(self, rew_batch_test, done_batch_test):

        steps = rew_batch_test.shape[0]
        rew_batch_test = rew_batch_test.T
        done_batch_test = done_batch_test.T
        for i in range(self.n_envs):
            for j in range(steps):
                self.episode_rewards_test[i].append(rew_batch_test[i][j])
                if done_batch_test[i][j]:
                    self.episode_len_buffer_test.append(len(self.episode_rewards_test[i]))
                    self.episode_reward_buffer_test.append(np.sum(self.episode_rewards_test[i]))
                    self.episode_rewards_test[i] = []
                    self.num_episodes_test += 1

    def feed_train(self, rew_batch, done_batch):
        steps = rew_batch.shape[0]
        n_envs = rew_batch.shape[1]
        rew_batch = rew_batch.T
        done_batch = done_batch.T

        for i in range(n_envs):
            for j in range(steps):
                self.episode_rewards[i].append(rew_batch[i][j])
                if done_batch[i][j]:
                    self.episode_len_buffer.append(len(self.episode_rewards[i]))
                    self.episode_reward_buffer.append(np.sum(self.episode_rewards[i]))
                    self.episode_rewards[i] = []
                    self.num_episodes += 1



    def get_episode_statistics(self):
        episode_statistics = {}
        episode_statistics['Rewards/max_episodes']  = {'train': np.max(self.episode_reward_buffer),
                                                       'test':np.max(self.episode_reward_buffer_test)}
        episode_statistics['Rewards/mean_episodes'] = {'train': np.mean(self.episode_reward_buffer),
                                                       'test': np.mean(self.episode_reward_buffer_test)}
        episode_statistics['Rewards/min_episodes']  = {'train': np.min(self.episode_reward_buffer),
                                                       'test': np.min(self.episode_reward_buffer_test)}
        episode_statistics['Rewards/median_episodes']  = {'train': np.median(self.episode_reward_buffer),
                                                       'test': np.median(self.episode_reward_buffer_test)}

        episode_statistics['Len/max_episodes']  = {'train': np.max(self.episode_len_buffer),
                                                   'test': np.max(self.episode_len_buffer_test)}
        episode_statistics['Len/mean_episodes'] = {'train': np.mean(self.episode_len_buffer),
                                                   'test': np.mean(self.episode_len_buffer_test)}
        episode_statistics['Len/min_episodes']  = {'train': np.min(self.episode_len_buffer),
                                                   'test': np.min(self.episode_len_buffer_test)}

        return episode_statistics

    def get_train_statistics(self):
        train_statistics = {}
        train_statistics['Rewards_max_episodes'] = np.max(self.episode_reward_buffer)

        train_statistics['Rewards_mean_episodes'] = np.mean(self.episode_reward_buffer)

        train_statistics['Rewards_median_episodes'] = np.median(self.episode_reward_buffer)

        train_statistics['Rewards_min_episodes'] = np.min(self.episode_reward_buffer)

        train_statistics['Len_max_episodes'] = np.max(self.episode_len_buffer)

        train_statistics['Len_mean_episodes'] = np.mean(self.episode_len_buffer)

        train_statistics['Len_min_episodes'] = np.min(self.episode_len_buffer)


        return train_statistics


class maxEnt_Logger(Logger):
    def __init__(self, n_envs, max_reward_seeds, start_train_test, obs_shape, obs_full_shape, recurrent_hidden_state_size, device='cpu'):
        super(maxEnt_Logger, self).__init__(n_envs, obs_shape, obs_full_shape, recurrent_hidden_state_size, device)


        self.obs_full = {}
        self.obs_full['train_eval'] = torch.zeros(self.n_envs, *obs_full_shape)
        self.obs_full['test_eval'] = torch.zeros(self.n_envs, *obs_full_shape)

        self.obs_vec_ds = {}
        self.obs_vec_ds['train_eval'] = []
        for i in range(n_envs):
            self.obs_vec_ds['train_eval'].append([])

        self.obs_vec_ds['test_eval'] = []
        for i in range(n_envs):
            self.obs_vec_ds['test_eval'].append([])

        self.max_reward_seeds = max_reward_seeds
        self.start_train_test = start_train_test

        self.episode_rewards_train_ext = []
        self.episode_rewards_test_ext = []
        for _ in range(self.n_envs):
            self.episode_rewards_train_ext.append([])
            self.episode_rewards_test_ext.append([])

        self.episode_reward_buffer_train_vs_oracle = deque(maxlen=self.n_envs)
        self.episode_reward_buffer_train_completed = deque(maxlen=self.n_envs)
        self.episode_reward_buffer_test_vs_oracle  = deque(maxlen=self.n_envs)
        self.episode_reward_buffer_test_completed  = deque(maxlen=self.n_envs)
        self.episode_reward_buffer_train_ext       = deque(maxlen=self.n_envs)
        self.episode_reward_buffer_test_ext        = deque(maxlen=self.n_envs)

    def feed_eval_test(self, rew_batch_train, done_batch_train, rew_batch_train_ext, rew_batch_test, done_batch_test, rew_batch_test_ext,
                              seeds_batch_train, seeds_batch_test):


        steps = rew_batch_test.shape[0]
        rew_batch_train = rew_batch_train.T
        done_batch_train = done_batch_train.T
        rew_batch_train_ext = rew_batch_train_ext.T
        seeds_batch_train = seeds_batch_train.T

        rew_batch_test = rew_batch_test.T
        done_batch_test = done_batch_test.T
        rew_batch_test_ext = rew_batch_test_ext.T
        seeds_batch_test = seeds_batch_test.T


        for i in range(self.n_envs):
            for j in range(steps):
                self.episode_rewards_test[i].append(rew_batch_test[i][j])
                self.episode_rewards_test_ext[i].append(rew_batch_test_ext[i][j])
                self.episode_rewards_train[i].append(rew_batch_train[i][j])
                self.episode_rewards_train_ext[i].append(rew_batch_train_ext[i][j])
                if done_batch_train[i][j]:
                    train_seed = seeds_batch_train[i][j]
                    max_reward_train_seeds = self.max_reward_seeds['train_eval'][int(train_seed) - self.start_train_test['train_eval']]
                    self.episode_len_buffer_train.append(len(self.episode_rewards_train[i]))
                    self.episode_reward_buffer_train.append(np.sum(self.episode_rewards_train[i]))
                    self.episode_reward_buffer_train_ext.append(np.sum(self.episode_rewards_train_ext[i]))
                    train_vs_oracle = np.sum(self.episode_rewards_train[i]) / (max_reward_train_seeds + 1e-4)
                    if train_vs_oracle > 1:
                        # print(f"bug! train sum reward is {train_vs_oracle} seed {train_seed}")
                        train_vs_oracle = 1
                    self.episode_reward_buffer_train_vs_oracle.append(train_vs_oracle)
                    self.episode_reward_buffer_train_completed.append(1*(np.sum(self.episode_rewards_train[i]) == max_reward_train_seeds))
                    self.episode_rewards_train[i] = []
                    self.episode_rewards_train_ext[i] = []
                    self.num_episodes_train += 1

                if done_batch_test[i][j]:
                    test_seed = seeds_batch_test[i][j]
                    max_reward_test_seeds = self.max_reward_seeds['test_eval'][int(test_seed) - self.start_train_test['test_eval']]
                    self.episode_len_buffer_test.append(len(self.episode_rewards_test[i]))
                    self.episode_reward_buffer_test.append(np.sum(self.episode_rewards_test[i]))
                    self.episode_reward_buffer_test_ext.append(np.sum(self.episode_rewards_test_ext[i]))
                    test_vs_oracle = np.sum(self.episode_rewards_test[i]) / (max_reward_test_seeds + 1e-4)
                    if test_vs_oracle > 1:
                        # print(f"bug! test nondet sum reward is {test_vs_oracle_nondet} seed {test_seed_nondet}")
                        test_vs_oracle = 1
                    self.episode_reward_buffer_test_vs_oracle.append(test_vs_oracle)
                    self.episode_reward_buffer_test_completed.append(1*(np.sum(self.episode_rewards_test[i]) == max_reward_test_seeds))
                    self.episode_rewards_test[i] = []
                    self.episode_rewards_test_ext[i] = []
                    self.num_episodes_test += 1

    def get_episode_statistics(self):
        episode_statistics = {}
        episode_statistics['Rewards/max_episodes'] = {'train': np.max(self.episode_reward_buffer),
                                                      'train_eval': np.max(self.episode_reward_buffer_train),
                                                      'train_eval_ext': np.max(self.episode_reward_buffer_train_ext),
                                                      'train_eval_vs_oracle': np.max(self.episode_reward_buffer_train_vs_oracle),
                                                      'train_eval_completed': np.max(self.episode_reward_buffer_train_completed),
                                                      'test': np.max(self.episode_reward_buffer_test),
                                                      'test_ext': np.max(self.episode_reward_buffer_test_ext),
                                                      'test_vs_oracle': np.max(self.episode_reward_buffer_test_vs_oracle),
                                                      'test_completed': np.max(self.episode_reward_buffer_test_completed)}
        episode_statistics['Rewards/mean_episodes'] = {'train': np.mean(self.episode_reward_buffer),
                                                       'train_eval': np.mean(self.episode_reward_buffer_train),
                                                       'train_eval_ext': np.mean(self.episode_reward_buffer_train_ext),
                                                       'train_eval_vs_oracle': np.mean(self.episode_reward_buffer_train_vs_oracle),
                                                       'train_eval_completed': np.mean(self.episode_reward_buffer_train_completed),
                                                       'test': np.mean(self.episode_reward_buffer_test),
                                                       'test_ext': np.mean(self.episode_reward_buffer_test_ext),
                                                       'test_vs_oracle': np.mean(self.episode_reward_buffer_test_vs_oracle),
                                                       'test_completed': np.mean(self.episode_reward_buffer_test_completed)}
        episode_statistics['Rewards/min_episodes'] = {'train': np.min(self.episode_reward_buffer),
                                                      'train_eval': np.min(self.episode_reward_buffer_train),
                                                      'train_eval_ext': np.min(self.episode_reward_buffer_train_ext),
                                                      'train_eval_vs_oracle': np.min(self.episode_reward_buffer_train_vs_oracle),
                                                      'train_eval_completed': np.min(self.episode_reward_buffer_train_completed),
                                                      'test': np.min(self.episode_reward_buffer_test),
                                                      'test_ext': np.min(self.episode_reward_buffer_test_ext),
                                                      'test_vs_oracle': np.min(self.episode_reward_buffer_test_vs_oracle),
                                                      'test_completed': np.min(self.episode_reward_buffer_test_completed)}
        episode_statistics['Rewards/median_episodes']  = {'train': np.median(self.episode_reward_buffer),
                                                          'train_eval': np.median(self.episode_reward_buffer_train),
                                                          'train_eval_ext': np.median(self.episode_reward_buffer_train_ext),
                                                          'test': np.median(self.episode_reward_buffer_test),
                                                          'test_ext': np.median(self.episode_reward_buffer_test_ext),}

        episode_statistics['Len/max_episodes'] = {'train': np.max(self.episode_len_buffer),
                                                  'train_eval': np.max(self.episode_len_buffer_train),
                                                  'test': np.max(self.episode_len_buffer_test)}
        episode_statistics['Len/mean_episodes'] = {'train': np.mean(self.episode_len_buffer),
                                                   'train_eval': np.mean(self.episode_len_buffer_train),
                                                   'test': np.mean(self.episode_len_buffer_test)}
        episode_statistics['Len/min_episodes'] = {'train': np.min(self.episode_len_buffer),
                                                  'train_eval': np.min(self.episode_len_buffer_train),
                                                  'test': np.min(self.episode_len_buffer_test)}


        return episode_statistics