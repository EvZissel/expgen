import numpy as np
import torch

from PPO_maxEnt_LEEP.distributions import FixedCategorical
from torch import nn


def evaluate_procgen(actor_critic, eval_envs, env_name,
                     device, steps, logger, deterministic=True):
    rew_batch = []
    done_batch = []

    for t in range(steps):
        with torch.no_grad():
            _, action, _, dist_probs, eval_recurrent_hidden_states = actor_critic.act(
                logger.obs[env_name].float().to(device),
                logger.eval_recurrent_hidden_states[env_name],
                logger.eval_masks[env_name],
                deterministic=deterministic)

            # Observe reward and next obs
            next_obs, reward, done, infos = eval_envs.step(action.squeeze().cpu().numpy())
            logger.eval_masks[env_name] = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=device)
            logger.eval_recurrent_hidden_states[env_name] = eval_recurrent_hidden_states

            if 'env_reward' in infos[0]:
                rew_batch.append([info['env_reward'] for info in infos])
            else:
                rew_batch.append(reward)
            done_batch.append(done)


            logger.obs[env_name] = next_obs

    rew_batch = np.array(rew_batch)
    done_batch = np.array(done_batch)

    return rew_batch, done_batch


def maxEnt_oracle(obs_all, action):
    next_action = action.clone().detach()
    for i in range(len(action)):
        obs = obs_all[i].cpu().numpy()
        action_i = action[i]
        new_action_i = np.array([7])

        min_r = np.nonzero((obs[1] == 1))[0].min()
        max_r = np.nonzero((obs[1] == 1))[0].max()
        middle_r = int(min_r + (max_r - min_r + 1) / 2)

        min_c = np.nonzero((obs[1] == 1))[1].min()
        max_c = np.nonzero((obs[1] == 1))[1].max()
        middle_c = int(min_c + (max_c - min_c + 1) / 2)

        if action_i == 7:
            if (max_r + 1 < 64) and obs[0][max_r + 1, middle_c] == 0:
                new_action_i = np.array([3])
            elif (max_c + 1 < 64) and obs[0][middle_r, max_c + 1] == 0:
                new_action_i = np.array([7])
            elif (min_r - 1 > 0) and obs[0][min_r - 1, middle_c] == 0:
                new_action_i = np.array([5])
            else:
                new_action_i = np.array([1])
        elif action_i == 5:
            if (max_c + 1 < 64) and obs[0][middle_r, max_c + 1] == 0:
                new_action_i = np.array([7])
            elif (min_r - 1 > 0) and obs[0][min_r - 1, middle_c] == 0:
                new_action_i = np.array([5])
            elif (min_c - 1 > 0) and obs[0][middle_r, min_c - 1] == 0:
                new_action_i = np.array([1])
            else:
                new_action_i = np.array([3])
        elif action_i == 3:
            if (min_c - 1 > 0) and obs[0][middle_r, min_c - 1] == 0:
                new_action_i = np.array([1])
            elif (max_r + 1 < 64) and obs[0][max_r + 1, middle_c] == 0:
                new_action_i = np.array([3])
            elif (max_c + 1 < 64) and obs[0][middle_r, max_c + 1] == 0:
                new_action_i = np.array([7])
            else:
                new_action_i = np.array([5])
        elif action_i == 1:
            if (min_r - 1 > 0) and obs[0][min_r - 1, middle_c] == 0:
                new_action_i = np.array([5])
            elif (min_c - 1 > 0) and obs[0][middle_r, min_c - 1] == 0:
                new_action_i = np.array([1])
            elif (max_r + 1 < 64) and obs[0][max_r + 1, middle_c] == 0:
                new_action_i = np.array([3])
            else:
                new_action_i = np.array([7])

        next_action[i] = torch.tensor(new_action_i)

    return next_action


def evaluate_procgen_LEEP(actor_critic_0, actor_critic_1, actor_critic_2, actor_critic_3, eval_envs_dic, env_name,
                          device, steps, logger, deterministic=True, num_ensemble=4,
                          actor_critic_4=None, actor_critic_5=None, actor_critic_6=None, actor_critic_7=None, actor_critic_8=None, actor_critic_9=None):
    eval_envs = eval_envs_dic[env_name]
    rew_batch = []
    done_batch = []
    seed_batch = []

    for t in range(steps):
        with torch.no_grad():
            _, action0, _, dist_probs, eval_recurrent_hidden_states = actor_critic_0.act(
                logger.obs[env_name].float().to(device),
                logger.eval_recurrent_hidden_states[env_name],
                logger.eval_masks[env_name],
                deterministic=deterministic)

            _, action_1, _, dist_probs_1, eval_recurrent_hidden_states_1 = actor_critic_1.act(
                logger.obs[env_name].float().to(device),
                logger.eval_recurrent_hidden_states[env_name],
                logger.eval_masks[env_name],
                deterministic=deterministic)

            _, action_2, _, dist_probs_2, eval_recurrent_hidden_states_2 = actor_critic_2.act(
                logger.obs[env_name].float().to(device),
                logger.eval_recurrent_hidden_states[env_name],
                logger.eval_masks[env_name],
                deterministic=deterministic)

            _, action_3, _, dist_probs_3, eval_recurrent_hidden_states_3 = actor_critic_3.act(
                logger.obs[env_name].float().to(device),
                logger.eval_recurrent_hidden_states[env_name],
                logger.eval_masks[env_name],
                deterministic=deterministic)

            if num_ensemble > 4:
                _, action_4, _, dist_probs_4, eval_recurrent_hidden_states_4 = actor_critic_4.act(
                    logger.obs[env_name].float().to(device),
                    logger.eval_recurrent_hidden_states[env_name],
                    logger.eval_masks[env_name],
                    deterministic=deterministic)

                _, action_5, _, dist_probs_5, eval_recurrent_hidden_states_5 = actor_critic_5.act(
                    logger.obs[env_name].float().to(device),
                    logger.eval_recurrent_hidden_states[env_name],
                    logger.eval_masks[env_name],
                    deterministic=deterministic)

                if num_ensemble > 6:
                    _, action_6, _, dist_probs_6, eval_recurrent_hidden_states_6 = actor_critic_6.act(
                        logger.obs[env_name].float().to(device),
                        logger.eval_recurrent_hidden_states[env_name],
                        logger.eval_masks[env_name],
                        deterministic=deterministic)

                    _, action_7, _, dist_probs_7, eval_recurrent_hidden_states_7 = actor_critic_7.act(
                        logger.obs[env_name].float().to(device),
                        logger.eval_recurrent_hidden_states[env_name],
                        logger.eval_masks[env_name],
                        deterministic=deterministic)

                if num_ensemble > 8:
                    _, action_8, _, dist_probs_8, eval_recurrent_hidden_states_8 = actor_critic_8.act(
                        logger.obs[env_name].float().to(device),
                        logger.eval_recurrent_hidden_states[env_name],
                        logger.eval_masks[env_name],
                        deterministic=deterministic)

                    _, action_9, _, dist_probs_9, eval_recurrent_hidden_states_9 = actor_critic_9.act(
                        logger.obs[env_name].float().to(device),
                        logger.eval_recurrent_hidden_states[env_name],
                        logger.eval_masks[env_name],
                        deterministic=deterministic)

            max_policy = torch.max(torch.max(torch.max(dist_probs, dist_probs_1), dist_probs_2), dist_probs_3)
            if num_ensemble > 4:
                max_policy = torch.max(torch.max(max_policy, dist_probs_4), dist_probs_5)
            if num_ensemble > 6:
                max_policy = torch.max(torch.max(max_policy, dist_probs_6), dist_probs_7)
            if num_ensemble > 8:
                max_policy = torch.max(torch.max(max_policy, dist_probs_8), dist_probs_9)
            max_policy = torch.div(max_policy, max_policy.sum(1).unsqueeze(1))

            if deterministic:
                action = max_policy.max(1)[1]
            else:
                x = FixedCategorical(logits=max_policy)
                action = x.sample()
            # Observe reward and next obs
            next_obs, reward, done, infos = eval_envs.step(action.squeeze().cpu().numpy())
            logger.eval_masks[env_name] = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=device)
            logger.eval_recurrent_hidden_states[env_name] = eval_recurrent_hidden_states

            if t == 0:
                prev_seeds = np.zeros_like(reward)
                for i in range(len(done)):
                    prev_seeds[i] = infos[i]['prev_level_seed']
                seed_batch.append(prev_seeds)

            seeds = np.zeros_like(reward)
            for i in range(len(done)):
                seeds[i] = infos[i]['level_seed']
                if done[i] == 1:
                    logger.obs_sum[env_name][i] = next_obs[i].cpu()

            next_obs_sum = logger.obs_sum[env_name] + next_obs

            rew_batch.append(reward)
            done_batch.append(done)
            seed_batch.append(seeds)

            logger.obs[env_name] = next_obs
            logger.obs_sum[env_name] = next_obs_sum

    rew_batch = np.array(rew_batch)
    done_batch = np.array(done_batch)
    seed_batch = np.array(seed_batch)

    return rew_batch, done_batch, seed_batch


def evaluate_procgen_ensemble(actor_critic, actor_critic_1, actor_critic_2, actor_critic_3, actor_critic_4, actor_critic_5, actor_critic_6, actor_critic_7, actor_critic_8, actor_critic_9, actor_critic_maxEnt,
                              eval_envs_dic, env_name, num_processes,
                              device, steps, logger, deterministic=True,
                              num_detEnt=0, rand_act=False,num_ensemble=4, num_agree=4 ,maze_miner=False, num_agent=0):
    eval_envs = eval_envs_dic[env_name]
    rew_batch = []
    done_batch = []
    seed_batch = []

    m = FixedCategorical(
        torch.tensor([0.55, 0.25, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025]).repeat(num_processes, 1)) #worked for maze
    rand_action = FixedCategorical(torch.tensor(
        [0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067,
         1 - 14 * 0.067]).repeat(num_processes, 1))
    maxEnt_steps = torch.zeros(num_processes, 1, device=device)
    for t in range(steps):
        with torch.no_grad():
            _, action0, _, dist_probs, eval_recurrent_hidden_states = actor_critic.act(
                logger.obs[env_name].float().to(device),
                torch.zeros(num_processes, actor_critic.recurrent_hidden_state_size, device=device),
                logger.eval_masks[env_name],
                deterministic=deterministic)

            _, action1, _, dist_probs1, eval_recurrent_hidden_states1 = actor_critic_1.act(
                logger.obs[env_name].float().to(device),
                torch.zeros(num_processes, actor_critic.recurrent_hidden_state_size, device=device),
                logger.eval_masks[env_name],
                deterministic=deterministic)

            _, action2, _, dist_probs2, eval_recurrent_hidden_states2 = actor_critic_2.act(
                logger.obs[env_name].float().to(device),
                torch.zeros(num_processes, actor_critic.recurrent_hidden_state_size, device=device),
                logger.eval_masks[env_name],
                deterministic=deterministic)

            _, action3, _, dist_probs3, eval_recurrent_hidden_states3 = actor_critic_3.act(
                logger.obs[env_name].float().to(device),
                torch.zeros(num_processes, actor_critic.recurrent_hidden_state_size, device=device),
                logger.eval_masks[env_name],
                deterministic=deterministic)


            _, action_maxEnt, _, dist_probs_maxEnt, eval_recurrent_hidden_states_maxEnt = actor_critic_maxEnt.act(
                logger.obs[env_name].float().to(device),
                logger.eval_recurrent_hidden_states_maxEnt[env_name],
                logger.eval_masks[env_name],
                deterministic=deterministic)

            if num_ensemble > 4:
                _, action4, _, dist_probs4, eval_recurrent_hidden_states4 = actor_critic_4.act(
                    logger.obs[env_name].float().to(device),
                    torch.zeros(num_processes, actor_critic.recurrent_hidden_state_size, device=device),
                    logger.eval_masks[env_name],
                    deterministic=deterministic)

                _, action5, _, dist_probs5, eval_recurrent_hidden_states5 = actor_critic_5.act(
                    logger.obs[env_name].float().to(device),
                    torch.zeros(num_processes, actor_critic.recurrent_hidden_state_size, device=device),
                    logger.eval_masks[env_name],
                    deterministic=deterministic)

            if num_ensemble > 6:
                _, action6, _, dist_probs6, eval_recurrent_hidden_states6 = actor_critic_6.act(
                    logger.obs[env_name].float().to(device),
                    torch.zeros(num_processes, actor_critic.recurrent_hidden_state_size, device=device),
                    logger.eval_masks[env_name],
                    deterministic=deterministic)

                _, action7, _, dist_probs7, eval_recurrent_hidden_states7= actor_critic_7.act(
                    logger.obs[env_name].float().to(device),
                    torch.zeros(num_processes, actor_critic.recurrent_hidden_state_size, device=device),
                    logger.eval_masks[env_name],
                    deterministic=deterministic)

            if num_ensemble > 8:
                _, action8, _, dist_probs8, eval_recurrent_hidden_states8 = actor_critic_8.act(
                    logger.obs[env_name].float().to(device),
                    torch.zeros(num_processes, actor_critic.recurrent_hidden_state_size, device=device),
                    logger.eval_masks[env_name],
                    deterministic=deterministic)

                _, action9, _, dist_probs9, eval_recurrent_hidden_states9 = actor_critic_9.act(
                    logger.obs[env_name].float().to(device),
                    torch.zeros(num_processes, actor_critic.recurrent_hidden_state_size, device=device),
                    logger.eval_masks[env_name],
                    deterministic=deterministic)

            actions = []
            actions.append(action0)
            actions.append(action1)
            actions.append(action2)
            actions.append(action3)
            cardinal_left = 1*(action0 == 0)+1*(action0 == 1) + 1*(action0 == 2) + 1*(action1 == 0)+1*(action1 == 1) + 1*(action1 == 2) + 1*(action2 == 0)+1*(action2 == 1) + 1*(action2 == 2)\
                            + 1 * (action3 == 0) + 1 * (action3 == 1) + 1 * (action3 == 2)
            cardinal_right  = 1*(action0 == 6)+1*(action0 == 7) + 1*(action0 == 8) + 1*(action1 == 6)+1*(action1 == 7) + 1*(action1 == 8) + 1*(action2 == 6)+1*(action2 == 7) + 1*(action2 == 8)\
                            + 1 * (action3 == 6) + 1 * (action3 == 7) + 1 * (action3 == 8)
            if (maze_miner):  #Maze and Miner do not have right down/up left down/up
                cardinal_down = 1 * (action0 == 3) + 1 * (action1 == 3) + 1 * (action2 == 3) + 1 * (action3 == 3)
                cardinal_up = 1 * (action0 == 5) + 1 * (action1 == 5) + 1 * (action2 == 5) + 1 * (action3 == 5)
            else:
                cardinal_down  = 1*(action0 == 3) + 1*(action1 == 3) + 1*(action2 == 3) + 1*(action3 == 3) + 1*(action0 == 0) + 1*(action1 == 0) + 1*(action2 == 0) + 1*(action3 == 0)\
                                + 1*(action0 == 6) + 1*(action1 == 6) + 1*(action2 == 6) + 1*(action3 == 6)
                cardinal_up  = 1*(action0 == 5) + 1*(action1 == 5) + 1*(action2 == 5) + 1*(action3 == 5) + 1*(action0 == 2) + 1*(action1 == 2) + 1*(action2 == 2) + 1*(action3 == 2) \
                               + 1 * (action0 == 8) + 1 * (action1 == 8) + 1 * (action2 == 8) + 1 * (action3 == 8)
                cardinal_fire  = 1*(action0 == 9) + 1*(action1 == 9) + 1*(action2 == 9) + 1*(action3 == 9)
                cardinal_else  = 1*(action0 == 4) + 1*(action0 == 10) + 1*(action0 == 11) + 1*(action0 == 12) + 1*(action0 == 13) + 1*(action0 == 14) \
                               + 1*(action1 == 9) + 1*(action1 == 10) + 1*(action1 == 11) + 1*(action1 == 12) + 1*(action1 == 13) + 1*(action1 == 14)  \
                               + 1*(action2 == 9) + 1*(action2 == 10) + 1*(action2 == 11) + 1*(action2 == 12) + 1*(action2 == 13) + 1*(action2 == 14)  \
                               + 1*(action3 == 9) + 1*(action3 == 10) + 1*(action3 == 11) + 1*(action3 == 12) + 1*(action3 == 13) + 1*(action3 == 14)


            if num_ensemble > 4:
                actions.append(action4)
                actions.append(action5)
                cardinal_left += 1 * (action4 == 0) + 1 * (action4 == 1) + 1 * (action4 == 2) + 1 * (action5 == 0) + 1 * (action5 == 1) + 1 * (action5 == 2)
                cardinal_right += 1 * (action4 == 6) + 1 * (action4 == 7) + 1 * (action4 == 8) + 1 * (action5 == 6) + 1 * (action5 == 7) + 1 * (action5 == 8)
                if (maze_miner):
                    cardinal_down += 1 * (action4 == 3) + 1 * (action5 == 3)
                    cardinal_up += 1 * (action4 == 5) + 1 * (action5 == 5)
                else:
                    cardinal_down += 1 * (action4 == 3) + 1 * (action5 == 3) + 1 * (action4 == 0) + 1 * (action5 == 0) + 1 * (action4 == 6) + 1 * (action5 == 6)
                    cardinal_up += 1 * (action4 == 5) + 1 * (action5 == 5) + 1 * (action4 == 2) + 1 * (action5 == 2) + 1 * (action4 == 8) + 1 * (action5 == 8)
                    cardinal_fire += 1 * (action4 == 9) + 1 * (action5 == 9)
                    cardinal_else += 1 * (action4 == 4) + 1 * (action4 == 10) + 1 * (action4 == 11) + 1 * (action4 == 12) + 1 * (action4 == 13) + 1 * (action4 == 14) \
                                  + 1 * (action5 == 9) + 1 * (action5 == 10) + 1 * (action5 == 11) + 1 * (action5 == 12) + 1 * (action5 == 13) + 1 * (action5 == 14)


            if num_ensemble > 6:
                actions.append(action6)
                actions.append(action7)
                cardinal_left += 1 * (action6 == 0) + 1 * (action6 == 1) + 1 * (action6 == 2) + 1 * (action7 == 0) + 1 * (action7 == 1) + 1 * (action7 == 2)
                cardinal_right += 1 * (action6 == 6) + 1 * (action6 == 7) + 1 * (action6 == 8) + 1 * (action7 == 6) + 1 * (action7 == 7) + 1 * (action7 == 8)
                if (maze_miner):
                    cardinal_down += 1 * (action6 == 3) + 1 * (action7 == 3)
                    cardinal_up += 1 * (action6 == 5) + 1 * (action7 == 5)
                else:
                    cardinal_down += 1 * (action6 == 3) + 1 * (action7 == 3) + 1 * (action6 == 0) + 1 * (action7 == 0) + 1 * (action6 == 6) + 1 * (action7 == 6)
                    cardinal_up += 1 * (action6 == 5) + 1 * (action7 == 5) + 1 * (action6 == 2) + 1 * (action7 == 2) + 1 * (action6 == 8) + 1 * (action7 == 8)
                    cardinal_fire += 1 * (action6 == 9) + 1 * (action7 == 9)
                    cardinal_else += 1 * (action6 == 4) + 1 * (action6 == 10) + 1 * (action6 == 11) + 1 * (action6 == 12) + 1 * (action6 == 13) + 1 * (action6 == 14) \
                                  + 1 * (action7 == 9) + 1 * (action7 == 10) + 1 * (action7 == 11) + 1 * (action7 == 12) + 1 * (action7 == 13) + 1 * (action7 == 14)

            if num_ensemble > 8:
                actions.append(action8)
                actions.append(action9)
                cardinal_left += 1 * (action8 == 0) + 1 * (action8 == 1) + 1 * (action8 == 2) + 1 * (action9 == 0) + 1 * (action9 == 1) + 1 * (action9 == 2)
                cardinal_right += 1 * (action8 == 6) + 1 * (action8 == 7) + 1 * (action8 == 8) + 1 * (action9 == 6) + 1 * (action9 == 7) + 1 * (action9 == 8)
                if (maze_miner):
                    cardinal_down += 1 * (action8 == 3) + 1 * (action9 == 3)
                    cardinal_up += 1 * (action8 == 5) + 1 * (action9 == 5)
                else:
                    cardinal_down += 1 * (action8 == 3) + 1 * (action9 == 3) + 1 * (action8 == 0) + 1 * (action9 == 0) + 1 * (action8 == 6) + 1 * (action9 == 6)
                    cardinal_up += 1 * (action8 == 5) + 1 * (action9 == 5) + 1 * (action8 == 2) + 1 * (action9 == 2) + 1 * (action8 == 8) + 1 * (action9 == 8)
                    cardinal_fire += 1 * (action8 == 9) + 1 * (action9 == 9)
                    cardinal_else += 1 * (action8 == 4) + 1 * (action8 == 10) + 1 * (action8 == 11) + 1 * (action8 == 12) + 1 * (action8 == 13) + 1 * (action8 == 14) \
                                  + 1 * (action9 == 9) + 1 * (action9 == 10) + 1 * (action9 == 11) + 1 * (action9 == 12) + 1 * (action9 == 13) + 1 * (action9 == 14)


            if (maze_miner):
                directions = torch.cat((cardinal_up, cardinal_right, cardinal_down, cardinal_left), dim=1)
            else:
                directions = torch.cat((cardinal_up, cardinal_right, cardinal_down, cardinal_left, cardinal_fire, cardinal_else),dim=1)


            action_cardinal_left = 1 * (actions[num_agent] == 0) + 1 * (actions[num_agent] == 1) + 1 * (actions[num_agent] == 2)
            action_cardinal_right = 1 * (actions[num_agent] == 6) + 1 * (actions[num_agent] == 7) + 1 * (actions[num_agent] == 8)
            if (maze_miner):
                action_cardinal_down = 1 * (actions[num_agent] == 3)
                action_cardinal_up = 1 * (actions[num_agent] == 5)
                action_directions = torch.cat((action_cardinal_up, action_cardinal_right, action_cardinal_down, action_cardinal_left), dim=1)
            else:
                action_cardinal_down = 1 * (actions[num_agent] == 3) + 1 * (actions[num_agent] == 0) + 1 * (actions[num_agent] == 6)
                action_cardinal_up = 1 * (actions[num_agent] == 5) + 1 * (actions[num_agent] == 2) + 1 * (actions[num_agent] == 8)
                action_cardinal_fire = 1 * (actions[num_agent] == 9)
                action_cardinal_else = 1 * (actions[num_agent] == 4) + 1 * (actions[num_agent] == 10) + 1 * (actions[num_agent] == 11) + 1 * (actions[num_agent] == 12) + 1 * (actions[num_agent] == 13) + 1 * (actions[num_agent] == 14)
                action_directions = torch.cat((action_cardinal_up, action_cardinal_right, action_cardinal_down, action_cardinal_left, action_cardinal_fire, action_cardinal_else), dim=1)

            action_cardinal_index = torch.max(action_directions, dim=1)[1]

            is_equal = (directions[torch.arange(32), action_cardinal_index] >= num_agree).unsqueeze(1)
            action_NN = actions[num_agent]

            maxEnt_steps = maxEnt_steps - 1

            maxEnt_steps_sample = (~is_equal)*(maxEnt_steps<=0)
            maxEnt_steps = (m.sample() + 1).to(device)*maxEnt_steps_sample + maxEnt_steps*(~maxEnt_steps_sample)

            is_action = is_equal*(maxEnt_steps<=0)


            if num_detEnt > 0:
                maxEnt_steps = (num_detEnt * torch.ones(num_processes, 1, device=device)).to(device)*maxEnt_steps_sample + maxEnt_steps*(~maxEnt_steps_sample)

            action = action_NN * is_action + action_maxEnt * (~is_action)

            if rand_act:
                action = action_NN * is_action + rand_action.sample().to(device) * (~is_action)

            next_obs, reward, done, infos = eval_envs.step(action.squeeze().cpu().numpy())
            logger.eval_masks[env_name] = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=device)
            # Only for recurrent networks
            logger.eval_recurrent_hidden_states_maxEnt[env_name] = eval_recurrent_hidden_states_maxEnt

            seeds = np.zeros_like(reward)
            for i in range(len(done)):
                seeds[i] = infos[i]['level_seed']

            seed_batch.append(seeds)
            rew_batch.append(reward)
            done_batch.append(done)

            logger.obs[env_name] = next_obs

    rew_batch = np.array(rew_batch)
    done_batch = np.array(done_batch)
    return rew_batch, done_batch

def evaluate_procgen_maxEnt_avepool_original_L2(actor_critic, eval_envs_dic, eval_envs_dic_full_obs, env_name,
                                             device, steps, logger, num_buffer, kernel_size=3, stride=3, deterministic=True, p_norm=2, neighbor_size=1):
    eval_envs = eval_envs_dic[env_name]
    eval_envs_full_obs = eval_envs_dic_full_obs[env_name]
    rew_batch = []
    int_rew_batch = []
    done_batch = []
    seed_batch = []
    down_sample_avg = nn.AvgPool2d(kernel_size, stride=stride)


    for t in range(steps):
        with torch.no_grad():
            _, action, _, dist_probs, eval_recurrent_hidden_states = actor_critic.act(
                logger.obs[env_name].float().to(device),
                logger.eval_recurrent_hidden_states[env_name],
                logger.eval_masks[env_name],
                deterministic=deterministic)


            # Observe reward and next obs
            next_obs, reward, done, infos = eval_envs.step(action.squeeze().cpu().numpy())
            next_obs_full, _, _, _ = eval_envs_full_obs.step(action.squeeze().cpu().numpy())

            logger.eval_masks[env_name] = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=device)
            logger.eval_recurrent_hidden_states[env_name] = eval_recurrent_hidden_states

            if t == 0:
                prev_seeds = np.zeros_like(reward)
                for i in range(len(done)):
                    prev_seeds[i] = infos[i]['prev_level_seed']
                seed_batch.append(prev_seeds)

            seeds = np.zeros_like(reward)
            int_reward = np.zeros_like(reward)
            next_obs_ds = down_sample_avg(next_obs_full)
            for i in range(len(done)):
                seeds[i] = infos[i]['level_seed']
                if done[i] == 1 :
                    logger.obs_vec_ds[env_name][i] = []
                else:
                    env_steps = len(logger.obs_vec_ds[env_name][i])
                    if env_steps > 0:
                        if env_steps > num_buffer:
                            old_obs = torch.stack(logger.obs_vec_ds[env_name][i][env_steps-num_buffer:])
                        else:
                            old_obs = torch.stack(logger.obs_vec_ds[env_name][i])
                        neighbor_size_i = int(min(neighbor_size, len(logger.obs_vec_ds[env_name][i])) -1)
                        int_reward[i]  = (old_obs - next_obs_ds[i].unsqueeze(0)).flatten(start_dim=1).norm(p=p_norm, dim=1).sort().values[neighbor_size_i]

                logger.obs_vec_ds[env_name][i].append(next_obs_ds[i])


            rew_batch.append(reward)
            int_rew_batch.append(int_reward)
            done_batch.append(done)
            seed_batch.append(seeds)

            logger.obs[env_name] = next_obs
            logger.obs_full[env_name] = next_obs_full
            logger.last_action[env_name] = action

    rew_batch = np.array(rew_batch)
    int_rew_batch = np.array(int_rew_batch)
    done_batch = np.array(done_batch)
    seed_batch = np.array(seed_batch)

    return rew_batch, int_rew_batch, done_batch, seed_batch