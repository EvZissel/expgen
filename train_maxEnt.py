import pathlib

from PPO_maxEnt_LEEP import algo, utils
from PPO_maxEnt_LEEP.arguments import get_args
from PPO_maxEnt_LEEP.envs import make_ProcgenEnvs
from PPO_maxEnt_LEEP.model import Policy, ImpalaModel
from PPO_maxEnt_LEEP.storage import RolloutStorage
from evaluation import evaluate_procgen_maxEnt_avepool_original_L2
from PPO_maxEnt_LEEP.procgen_wrappers import *
from PPO_maxEnt_LEEP.logger import maxEnt_Logger
import PPO_maxEnt_LEEP.hyperparams as hps
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

EVAL_ENVS = ['train_eval', 'test_eval']

def main():
    args = get_args()
    import random
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    logdir_ = args.env_name + '_ppo' + '_seed_' + str(args.seed)
    logdir_ = logdir_ + '_maxEnt'
    logdir = os.path.join(os.path.expanduser(args.log_dir), logdir_)
    utils.cleanup_log_dir(logdir)

    print("logdir: " + logdir)
    print("printing args")
    argslog = pd.DataFrame(columns=['args', 'value'])
    for key in vars(args):
        log = [key] + [vars(args)[key]]
        argslog.loc[len(argslog)] = log
        print(key, ':', vars(args)[key])

    with open(logdir + '/args.csv', 'w') as f:
        argslog.to_csv(f, index=False)

    progresslog = pd.DataFrame(columns=['timesteps', 'train intrinsic mean', 'train intrinsic min', 'train intrinsic max',
                                        'train extrinsic mean', 'train extrinsic min', 'train extrinsic max',
                                        'train vs oracle mean', 'train vs oracle min', 'train vs oracle max',
                                        'train completed mean', 'train completed min', 'train completed max',
                                        'test intrinsic mean', 'test intrinsic min', 'test intrinsic max',
                                        'test extrinsic mean', 'test extrinsic min', 'test extrinsic max',
                                        'test vs oracle mean', 'test vs oracle min', 'test vs oracle max',
                                        'test completed mean', 'test completed min', 'test completed max'])
    torch.set_num_threads(1)
    device = torch.device("cuda:{}".format(args.gpu_device) if args.cuda else "cpu")

    print('making envs...')
    max_reward_seeds = {
        'train_eval': [],
        'test_eval': []
    }

    test_start_level = args.start_level + args.num_level + 1
    start_train_test = {
        'train_eval': args.start_level,
        'test_eval': test_start_level
    }

    down_sample_avg = nn.AvgPool2d(args.kernel_size, stride=args.stride)

    # Calculate approximation of max reward per seed (only for L0)
    for eval_disp_name in EVAL_ENVS:
        for i in range(args.num_test_level):
            envs = make_ProcgenEnvs(num_envs=1,
                                    env_name=args.env_name,
                                    start_level=start_train_test[eval_disp_name] + i,
                                    num_levels=1,
                                    distribution_mode=args.distribution_mode,
                                    use_generated_assets=args.use_generated_assets,
                                    use_backgrounds=False,
                                    restrict_themes=args.restrict_themes,
                                    use_monochrome_assets=args.use_monochrome_assets,
                                    center_agent=False,
                                    rand_seed=args.seed,
                                    mask_size=args.mask_size,
                                    normalize_rew=args.normalize_rew,
                                    mask_all=args.mask_all)

            obs = envs.reset()
            obs = down_sample_avg(obs)
            reward = (obs[0][0] == 0).sum()
            max_reward_seeds[eval_disp_name].append(reward)

    envs = make_ProcgenEnvs(num_envs=args.num_processes,
                            env_name=args.env_name,
                            start_level=args.start_level,
                            num_levels=args.num_level,
                            distribution_mode=args.distribution_mode,
                            use_generated_assets=args.use_generated_assets,
                            use_backgrounds=args.use_backgrounds,
                            restrict_themes=args.restrict_themes,
                            use_monochrome_assets=args.use_monochrome_assets,
                            rand_seed=args.seed,
                            mask_size=args.mask_size,
                            normalize_rew=args.normalize_rew,
                            mask_all=args.mask_all,
                            device=device)

    envs_full_obs = make_ProcgenEnvs(num_envs=args.num_processes,
                                     env_name=args.env_name,
                                     start_level=args.start_level,
                                     num_levels=args.num_level,
                                     distribution_mode=args.distribution_mode,
                                     use_generated_assets=args.use_generated_assets,
                                     use_backgrounds=args.use_backgrounds,
                                     restrict_themes=args.restrict_themes,
                                     use_monochrome_assets=args.use_monochrome_assets,
                                     rand_seed=args.seed,
                                     center_agent=False,
                                     mask_size=args.mask_size,
                                     normalize_rew=args.normalize_rew,
                                     mask_all=args.mask_all,
                                     device=device)
    # Test envs
    eval_envs_dic = {}
    eval_envs_dic['train_eval'] = make_ProcgenEnvs(num_envs=args.num_processes,
                                                   env_name=args.env_name,
                                                   start_level=args.start_level,
                                                   num_levels=args.num_test_level,
                                                   distribution_mode=args.distribution_mode,
                                                   use_generated_assets=args.use_generated_assets,
                                                   use_backgrounds=args.use_backgrounds,
                                                   restrict_themes=args.restrict_themes,
                                                   use_monochrome_assets=args.use_monochrome_assets,
                                                   rand_seed=args.seed,
                                                   mask_size=args.mask_size,
                                                   normalize_rew=args.normalize_rew,
                                                   mask_all=args.mask_all,
                                                   device=device)

    eval_envs_dic['test_eval'] = make_ProcgenEnvs(num_envs=args.num_processes,
                                                  env_name=args.env_name,
                                                  start_level=test_start_level,
                                                  num_levels=args.num_test_level,
                                                  distribution_mode=args.distribution_mode,
                                                  use_generated_assets=args.use_generated_assets,
                                                  use_backgrounds=args.use_backgrounds,
                                                  restrict_themes=args.restrict_themes,
                                                  use_monochrome_assets=args.use_monochrome_assets,
                                                  rand_seed=args.seed,
                                                  mask_size=args.mask_size,
                                                  normalize_rew=args.normalize_rew,
                                                  mask_all=args.mask_all,
                                                  device=device)

    # Test envs full observation
    eval_envs_dic_full_obs = {}
    eval_envs_dic_full_obs['train_eval'] = make_ProcgenEnvs(num_envs=args.num_processes,
                                                            env_name=args.env_name,
                                                            start_level=args.start_level,
                                                            num_levels=args.num_test_level,
                                                            distribution_mode=args.distribution_mode,
                                                            use_generated_assets=args.use_generated_assets,
                                                            use_backgrounds=args.use_backgrounds,
                                                            restrict_themes=args.restrict_themes,
                                                            use_monochrome_assets=args.use_monochrome_assets,
                                                            rand_seed=args.seed,
                                                            center_agent=False,
                                                            mask_size=args.mask_size,
                                                            normalize_rew=args.normalize_rew,
                                                            mask_all=args.mask_all,
                                                            device=device)

    eval_envs_dic_full_obs['test_eval'] = make_ProcgenEnvs(num_envs=args.num_processes,
                                                           env_name=args.env_name,
                                                           start_level=test_start_level,
                                                           num_levels=args.num_test_level,
                                                           distribution_mode=args.distribution_mode,
                                                           use_generated_assets=args.use_generated_assets,
                                                           use_backgrounds=args.use_backgrounds,
                                                           restrict_themes=args.restrict_themes,
                                                           use_monochrome_assets=args.use_monochrome_assets,
                                                           rand_seed=args.seed,
                                                           center_agent=False,
                                                           mask_size=args.mask_size,
                                                           normalize_rew=args.normalize_rew,
                                                           mask_all=args.mask_all,
                                                           device=device)
    print('done')

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base=ImpalaModel,
        base_kwargs={'recurrent': True,
                     'hidden_size': args.recurrent_hidden_size, 'gray_scale': args.gray_scale},
        epsilon_RPO=args.epsilon_RPO)
    actor_critic.to(device)

    # Training agent
    agent = algo.PPO(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        num_tasks=args.num_processes,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay)

    # Rollout storage for agent
    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size, device=device)

    # Load previous model
    if (args.continue_from_epoch > 0) and args.save_dir != "":
        save_path = pathlib.Path(args.save_dir, args.env + '_ppo_seed_' + args.seed + '_maxEnt')
        actor_critic_weighs = torch.load(
            os.path.join(save_path, args.env_name + "-epoch-{}.pt".format(args.continue_from_epoch)),
            map_location=device)
        actor_critic.load_state_dict(actor_critic_weighs['state_dict'])
        agent.optimizer.load_state_dict(actor_critic_weighs['optimizer_state_dict'])

    logger = maxEnt_Logger(args.num_processes, max_reward_seeds, start_train_test, envs.observation_space.shape,
                           envs.observation_space.shape, actor_critic.recurrent_hidden_state_size, device=device)

    obs = envs.reset()
    obs_full = envs_full_obs.reset()
    obs_ds = down_sample_avg(obs_full)


    rollouts.obs[0].copy_(obs)
    rollouts.obs_ds[0].copy_(obs_ds)
    rollouts.obs_full.copy_(obs_full)
    rollouts.obs_sum.copy_(torch.zeros_like(obs_full))
    rollouts.obs0.copy_(obs_full)


    obs_train = eval_envs_dic['train_eval'].reset()
    logger.obs['train_eval'].copy_(obs_train)
    obs_train_full = eval_envs_dic_full_obs['train_eval'].reset()
    obs_train_ds = down_sample_avg(obs_train_full)
    for i in range(args.num_processes):
        logger.obs_vec_ds['train_eval'][i].append(obs_train_ds[i])
    logger.obs_full['train_eval'].copy_(obs_train_full)
    logger.obs_sum['train_eval'].copy_(torch.zeros_like(obs_train_full))
    logger.obs0['train_eval'].copy_(obs_train_full)

    obs_test = eval_envs_dic['test_eval'].reset()
    logger.obs['test_eval'].copy_(obs_test)
    obs_test_full = eval_envs_dic_full_obs['test_eval'].reset()
    obs_test_ds = down_sample_avg(obs_test_full)
    for i in range(args.num_processes):
        logger.obs_vec_ds['test_eval'][i].append(obs_test_ds[i])
    logger.obs_full['test_eval'].copy_(obs_test_full)
    logger.obs_sum['test_eval'].copy_(torch.zeros_like(obs_test_full))
    logger.obs0['test_eval'].copy_(obs_test_full)

    # Plot mazes
    fig = plt.figure(figsize=(20, 20))
    columns = 5
    rows = 5
    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(rollouts.obs[0][i].transpose(0, 2))
        plt.savefig(logdir + '/fig.png')

    seeds = torch.zeros(args.num_processes, 1)
    num_env_steps = hps.num_env_steps['maxEnt']
    num_updates = int(
        num_env_steps) // args.num_steps // args.num_processes


    for j in range(args.continue_from_epoch, args.continue_from_epoch + num_updates):

        # Policy rollouts
        actor_critic.eval()
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, _, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step].to(device), rollouts.recurrent_hidden_states[step].to(device),
                    rollouts.masks[step].to(device))

            # Observe reward and next obs
            obs, reward, done, infos = envs.step(action.squeeze().cpu().numpy())
            obs_full, reward_full, done_full, infos_full = envs_full_obs.step(action.squeeze().cpu().numpy())
            int_reward = np.zeros_like(reward)
            obs_ds = down_sample_avg(obs_full)


            diff_all = obs_ds.unsqueeze(0) - rollouts.obs_ds.to(device)

            for i in range(len(done)):
                if done[i] == 1:
                    rollouts.obs_sum[i] = torch.zeros_like(rollouts.obs_full[i])
                    rollouts.obs_full[i].copy_(obs_full[i])
                    rollouts.step_env[i] = 0
                else:
                    actual_step_env = int(max(0, rollouts.step_env[i] - args.num_buffer))

                    episode_start = int(step + 1 - rollouts.step_env[i] + actual_step_env)
                    diff = diff_all[max(0, episode_start):step+1][:, i, :, :]
                    if episode_start < 0:
                        if not len(diff):
                            diff = diff_all[args.num_steps + episode_start:args.num_steps][:, i, :, :]
                        else:
                            diff = torch.cat((diff, diff_all[args.num_steps + episode_start:args.num_steps][:, i, :, :].to(device)), dim=0)
                    if args.p_norm == 0:
                        diff = (1.0 * (diff.abs() > 1e-5)).sum(1)
                    neighbor_size = args.neighbor_size
                    if len(diff) < args.neighbor_size:
                        neighbor_size = len(diff)
                    int_reward[i] = diff.flatten(start_dim=1).norm(p=args.p_norm, dim=1).sort().values[int(neighbor_size-1)]


            for i, info in enumerate(infos):
                seeds[i] = info["level_seed"]
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, torch.from_numpy(int_reward).unsqueeze(1), masks, bad_masks, seeds, infos, obs_full)
            rollouts.obs_ds[rollouts.step].copy_(obs_ds)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1].to(device), rollouts.recurrent_hidden_states[-1].to(device),
                rollouts.masks[-1].to(device)).detach()

        actor_critic.train()
        gamma = hps.gamma[args.env_name]
        rollouts.compute_returns(next_value, use_gae=True, gamma=gamma, gae_lambda=args.gae_lambda)

        value_loss, action_loss, dist_entropy, _ = agent.update(rollouts)

        rollouts.after_update()

        rew_batch, done_batch = rollouts.fetch_log_data()
        logger.feed_train(rew_batch, done_batch[1:])

        # Save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0 or j == args.continue_from_epoch + num_updates - 1) and j > args.continue_from_epoch:
            torch.save({'state_dict': actor_critic.state_dict(), 'optimizer_state_dict': agent.optimizer.state_dict(),
                        'step': j}, os.path.join(logdir, args.env_name + "-epoch-{}.pt".format(j)))


        # Evaluate agent on evaluation tasks
        if ((args.eval_interval is not None and j % args.eval_interval == 0) or j == args.continue_from_epoch):
            actor_critic.eval()
            eval_dic_rew = {}
            eval_dic_int_rew = {}
            eval_dic_done = {}
            eval_dic_seeds = {}

            for eval_disp_name in EVAL_ENVS:
                eval_dic_rew[eval_disp_name], eval_dic_int_rew[eval_disp_name], eval_dic_done[eval_disp_name], \
                eval_dic_seeds[eval_disp_name] = evaluate_procgen_maxEnt_avepool_original_L2(actor_critic, eval_envs_dic,
                                                                                          eval_envs_dic_full_obs,
                                                                                          eval_disp_name, device,
                                                                                          args.num_steps, logger, args.num_buffer,
                                                                                          kernel_size=args.kernel_size,
                                                                                          stride=args.stride, deterministic=False, p_norm=args.p_norm, neighbor_size=args.neighbor_size)


            logger.feed_eval_test(eval_dic_int_rew['train_eval'], eval_dic_done['train_eval'], eval_dic_rew['train_eval'],
                                         eval_dic_int_rew['test_eval'], eval_dic_done['test_eval'], eval_dic_rew['test_eval'],
                                         eval_dic_seeds['train_eval'], eval_dic_seeds['test_eval'])

        # Print some stats
        if j % args.log_interval == 0:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            print('Iter {}, num timesteps {}, num training episodes {}, '
                  'dist_entropy {:.3f}, value_loss {:.3f}, action_loss {:.3f}\n'
                  .format(j, total_num_steps, logger.num_episodes, dist_entropy, value_loss, action_loss))
            episode_statistics = logger.get_episode_statistics()

            print(
                'Last {} training episodes: \n'
                'train mean/median intrinsic reward {:.1f}/{:.1f},\n'
                'train min/max intrinsic reward {:.1f}/{:.1f}\n'
                .format(args.num_processes,
                        episode_statistics['Rewards/mean_episodes']['train_eval'], episode_statistics['Rewards/median_episodes']['train_eval'],
                        episode_statistics['Rewards/min_episodes']['train_eval'], episode_statistics['Rewards/max_episodes']['train_eval']))

            print(
                'train mean/median extrinsic reward {:.1f}/{:.1f},\n'
                'train min/max extrinsic reward {:.1f}/{:.1f}\n'
                .format(episode_statistics['Rewards/mean_episodes']['train_eval_ext'], episode_statistics['Rewards/median_episodes']['train_eval_ext'],
                        episode_statistics['Rewards/min_episodes']['train_eval_ext'], episode_statistics['Rewards/max_episodes']['train_eval_ext']))
            print(
                'test mean/median intrinsic reward {:.1f}/{:.1f},\n'
                'test min/max intrinsic reward {:.1f}/{:.1f}\n'
                .format(episode_statistics['Rewards/mean_episodes']['test'], episode_statistics['Rewards/median_episodes']['test'],
                        episode_statistics['Rewards/min_episodes']['test'], episode_statistics['Rewards/max_episodes']['test']))

            print(
                'test mean/median extrinsic reward {:.1f}/{:.1f},\n'
                'test min/max extrinsic reward {:.1f}/{:.1f}\n'
                .format(episode_statistics['Rewards/mean_episodes']['test_ext'], episode_statistics['Rewards/median_episodes']['test_ext'],
                        episode_statistics['Rewards/min_episodes']['test_ext'], episode_statistics['Rewards/max_episodes']['test_ext']))


            log = [total_num_steps] + [episode_statistics['Rewards/mean_episodes']['train_eval']] + [episode_statistics['Rewards/min_episodes']['train_eval']] + [episode_statistics['Rewards/max_episodes']['train_eval']]
            log += [episode_statistics['Rewards/mean_episodes']['train_eval_ext']] + [episode_statistics['Rewards/min_episodes']['train_eval_ext']] + [episode_statistics['Rewards/max_episodes']['train_eval_ext']]
            log += [episode_statistics['Rewards/mean_episodes']['train_eval_vs_oracle']] + [episode_statistics['Rewards/min_episodes']['train_eval_vs_oracle']] + [episode_statistics['Rewards/max_episodes']['train_eval_vs_oracle']]
            log += [episode_statistics['Rewards/mean_episodes']['train_eval_completed']] + [episode_statistics['Rewards/min_episodes']['train_eval_completed']] + [episode_statistics['Rewards/max_episodes']['train_eval_completed']]
            log += [episode_statistics['Rewards/mean_episodes']['test']] + [episode_statistics['Rewards/min_episodes']['test']] + [episode_statistics['Rewards/max_episodes']['test']]
            log += [episode_statistics['Rewards/mean_episodes']['test_ext']] + [episode_statistics['Rewards/min_episodes']['test_ext']] + [episode_statistics['Rewards/max_episodes']['test_ext']]
            log += [episode_statistics['Rewards/mean_episodes']['test_vs_oracle']] + [episode_statistics['Rewards/min_episodes']['test_vs_oracle']] + [episode_statistics['Rewards/max_episodes']['test_vs_oracle']]
            log += [episode_statistics['Rewards/mean_episodes']['test_completed']] + [episode_statistics['Rewards/min_episodes']['test_completed']] + [episode_statistics['Rewards/max_episodes']['test_completed']]
            progresslog.loc[len(progresslog)] = log

            with open(logdir + '/progress_{}_seed_{}.csv'.format(args.env_name, args.seed), 'w') as f:
                progresslog.to_csv(f, index=False)

    # Training done. Save and clean up
    envs.close()
    for eval_disp_name in EVAL_ENVS:
        eval_envs_dic[eval_disp_name].close()


if __name__ == "__main__":
    main()
