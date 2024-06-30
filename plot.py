import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
plt.rcParams.update({'font.size': 23})


import pathlib

## Plot for PPO and Ensemble PPO (ExpGen)
env = 'maze'
seed = '0'
logdir = './logs'
# filedir = pathlib.Path(logdir,env+'_ppo_seed_'+ seed + '_ensemble')
filedir = pathlib.Path(logdir,env+'_ppo_seed_'+ seed + '_mask_all')
file = pathlib.Path(filedir,'progress_'+env+'_seed_'+seed+'.csv')

fig, axs = plt.subplots(nrows=2, ncols=3,figsize=(16, 6), dpi=300)

step = []
train_mean_episode_reward_raw = []
train_min_episode_reward_raw = []
train_max_episode_reward_raw = []
test_mean_episode_reward_raw = []
test_min_episode_reward_raw = []
test_max_episode_reward_raw = []


with open(file, 'r') as csv_file:
    reader = csv.reader(csv_file)

    for (i, row) in enumerate(reader):
        print(row)
        if i > 0:
            step.append(int(float(row[0])))
            train_mean_episode_reward_raw.append(float(row[1]))
            train_min_episode_reward_raw.append(float(row[2]))
            train_max_episode_reward_raw.append(float(row[3]))
            test_mean_episode_reward_raw.append(float(row[4]))
            test_min_episode_reward_raw.append(float(row[5]))
            test_max_episode_reward_raw.append(float(row[6]))

step = np.array(step)

train_mean_episode_reward = pd.DataFrame(train_mean_episode_reward_raw)
train_min_episode_reward = pd.DataFrame(train_min_episode_reward_raw)
train_max_episode_reward = pd.DataFrame(train_max_episode_reward_raw)
test_mean_episode_reward = pd.DataFrame(test_mean_episode_reward_raw)
test_min_episode_reward = pd.DataFrame(test_min_episode_reward_raw)
test_max_episode_reward = pd.DataFrame(test_max_episode_reward_raw)

train_mean_episode_reward = train_mean_episode_reward.ewm(alpha=0.2)
train_min_episode_reward = train_min_episode_reward.ewm(alpha=0.2)
train_max_episode_reward = train_max_episode_reward.ewm(alpha=0.2)
test_mean_episode_reward = test_mean_episode_reward.ewm(alpha=0.2)
test_min_episode_reward = test_min_episode_reward.ewm(alpha=0.2)
test_max_episode_reward = test_max_episode_reward.ewm(alpha=0.2)

train_mean_episode_reward = train_mean_episode_reward.mean().values.squeeze()
train_min_episode_reward = train_min_episode_reward.mean().values.squeeze()
train_max_episode_reward = train_max_episode_reward.mean().values.squeeze()
test_mean_episode_reward = test_mean_episode_reward.mean().values.squeeze()
test_min_episode_reward = test_min_episode_reward.mean().values.squeeze()
test_max_episode_reward = test_max_episode_reward.mean().values.squeeze()

axs[0][0].plot(step,test_mean_episode_reward, color=f'C{1}', alpha=0.2)
axs[0][1].plot(step,test_min_episode_reward, color=f'C{1}', alpha=0.2)
axs[0][2].plot(step,test_max_episode_reward, color=f'C{1}', alpha=0.2)
axs[1][0].plot(step,train_mean_episode_reward, color=f'C{2}', alpha=0.2)
axs[1][1].plot(step,train_min_episode_reward, color=f'C{2}', alpha=0.2)
axs[1][2].plot(step,train_max_episode_reward, color=f'C{2}', alpha=0.2)

axs[0][0].set_title('Test Mean')
axs[0][1].set_title('Test Min')
axs[0][2].set_title('Test Max')
axs[1][0].set_title('Train Mean')
axs[1][1].set_title('Train Min')
axs[1][2].set_title('Train Max')

axs[0][0].set_ylabel('Score')
axs[1][0].set_ylabel('Score')
axs[1][0].set_xlabel('step')
axs[1][1].set_xlabel('step')
axs[1][2].set_xlabel('step')
plt.show()


