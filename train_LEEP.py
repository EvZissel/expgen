import pathlib

from PPO_maxEnt_LEEP import algo, utils
from PPO_maxEnt_LEEP.arguments import get_args
from PPO_maxEnt_LEEP.envs import make_ProcgenEnvs
from PPO_maxEnt_LEEP.model import Policy, ImpalaModel_finetune
from PPO_maxEnt_LEEP.storage import RolloutStorage
from evaluation import  evaluate_procgen_LEEP
from PPO_maxEnt_LEEP.procgen_wrappers import *
from PPO_maxEnt_LEEP.logger import Logger
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch


EVAL_ENVS = ['train_eval','test_eval']


def main():
    args = get_args()
    import random; random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    logdir_ = args.env_name + '_LEEP_seed_' + str(args.seed)

    logdir = os.path.join(os.path.expanduser(args.log_dir), logdir_)
    utils.cleanup_log_dir(logdir)

    argslog = pd.DataFrame(columns=['args', 'value'])
    for key in vars(args):
        log = [key] + [vars(args)[key]]
        argslog.loc[len(argslog)] = log

    print("logdir: " + logdir)
    for key in vars(args):
        print(key, ':', vars(args)[key])

    with open(logdir + '/args.csv', 'w') as f:
        argslog.to_csv(f, index=False)

    progresslog = pd.DataFrame(columns=['timesteps', 'train mean', 'train min', 'train max', 'test mean', 'test min', 'test max'])
    torch.set_num_threads(1)
    device = torch.device("cuda:{}".format(args.gpu_device) if args.cuda else "cpu")

    print('making envs...')

    # Train envs
    num_envs = int(args.num_level/args.num_c)
    envs_dic = []
    for i in range(args.num_c):
        envs_dic.append(make_ProcgenEnvs(num_envs=int(args.num_processes/args.num_c),
                                         env_name=args.env_name,
                                         start_level=args.start_level + i*num_envs,
                                         num_levels=(i+1)*num_envs,
                                         distribution_mode=args.distribution_mode,
                                         use_generated_assets=args.use_generated_assets,
                                         use_backgrounds=args.use_backgrounds,
                                         restrict_themes=args.restrict_themes,
                                         use_monochrome_assets=args.use_monochrome_assets,
                                         rand_seed=args.seed,
                                         mask_size=args.mask_size,
                                         normalize_rew=args.normalize_rew,
                                         mask_all=args.mask_all,
                                         device=device))
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

    test_start_level = args.start_level + args.num_level + 1
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
    print('done')

    actor_critic_0 = Policy(
        envs_dic[0].observation_space.shape,
        envs_dic[0].action_space,
        base=ImpalaModel_finetune,
        base_kwargs={'recurrent': args.recurrent_policy or args.obs_recurrent,'hidden_size': args.recurrent_hidden_size})
    actor_critic_0.to(device)

    actor_critic_1 = Policy(
        envs_dic[1].observation_space.shape,
        envs_dic[1].action_space,
        base=ImpalaModel_finetune,
        base_kwargs={'recurrent': args.recurrent_policy or args.obs_recurrent,'hidden_size': args.recurrent_hidden_size})
    actor_critic_1.to(device)

    actor_critic_2 = Policy(
        envs_dic[2].observation_space.shape,
        envs_dic[2].action_space,
        base=ImpalaModel_finetune,
        base_kwargs={'recurrent': args.recurrent_policy or args.obs_recurrent,'hidden_size': args.recurrent_hidden_size})
    actor_critic_2.to(device)

    actor_critic_3 = Policy(
        envs_dic[3].observation_space.shape,
        envs_dic[3].action_space,
        base=ImpalaModel_finetune,
        base_kwargs={'recurrent': args.recurrent_policy or args.obs_recurrent,'hidden_size': args.recurrent_hidden_size})
    actor_critic_3.to(device)

    # Training agent 0
    agent_0 = algo.PPO_LEEP(
        actor_critic_0,
        actor_critic_1,
        actor_critic_2,
        actor_critic_3,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        args.KL_coef,
        lr=args.lr,
        eps=args.eps,
        num_tasks=int(args.num_processes/args.num_c),
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay)

    # Training agent 1
    agent_1 = algo.PPO_LEEP(
        actor_critic_1,
        actor_critic_2,
        actor_critic_3,
        actor_critic_0,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        args.KL_coef,
        lr=args.lr,
        eps=args.eps,
        num_tasks=int(args.num_processes/args.num_c),
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay)

    # Training agent 2
    agent_2 = algo.PPO_LEEP(
        actor_critic_2,
        actor_critic_3,
        actor_critic_0,
        actor_critic_1,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        args.KL_coef,
        lr=args.lr,
        eps=args.eps,
        num_tasks=int(args.num_processes/args.num_c),
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay)

    # Training agent 3
    agent_3 = algo.PPO_LEEP(
        actor_critic_3,
        actor_critic_0,
        actor_critic_1,
        actor_critic_2,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        args.KL_coef,
        lr=args.lr,
        eps=args.eps,
        num_tasks=int(args.num_processes/args.num_c),
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay)

    # Rollout storage for agents
    rollouts_0 = RolloutStorage(args.num_steps, int(args.num_processes/args.num_c),
                              envs_dic[0].observation_space.shape, envs_dic[0].observation_space.shape, envs_dic[0].action_space,
                              actor_critic_0.recurrent_hidden_state_size, device=device)
    rollouts_1 = RolloutStorage(args.num_steps, int(args.num_processes/args.num_c),
                              envs_dic[1].observation_space.shape, envs_dic[1].observation_space.shape, envs_dic[1].action_space,
                              actor_critic_1.recurrent_hidden_state_size, device=device)
    rollouts_2 = RolloutStorage(args.num_steps, int(args.num_processes/args.num_c),
                              envs_dic[2].observation_space.shape, envs_dic[2].observation_space.shape, envs_dic[2].action_space,
                              actor_critic_2.recurrent_hidden_state_size, device=device)
    rollouts_3 = RolloutStorage(args.num_steps, int(args.num_processes/args.num_c),
                              envs_dic[3].observation_space.shape, envs_dic[3].observation_space.shape, envs_dic[3].action_space,
                              actor_critic_3.recurrent_hidden_state_size, device=device)


    # Load previous model
    if (args.continue_from_epoch > 0) and args.save_dir != "":
        save_path = pathlib.Path(args.save_dir, args.env + '_LEEP_seed_' + args.seed)
        actor_critic_weighs = torch.load(os.path.join(save_path, args.env_name + "-epoch-{}.pt".format(args.continue_from_epoch)), map_location=device)
        actor_critic_0.load_state_dict(actor_critic_weighs['state_dict_0'])
        agent_0.optimizer.load_state_dict(actor_critic_weighs['optimizer_state_dict_0'])
        actor_critic_1.load_state_dict(actor_critic_weighs['state_dict_1'])
        agent_1.optimizer.load_state_dict(actor_critic_weighs['optimizer_state_dict_1'])
        actor_critic_2.load_state_dict(actor_critic_weighs['state_dict_2'])
        agent_2.optimizer.load_state_dict(actor_critic_weighs['optimizer_state_dict_2'])
        actor_critic_3.load_state_dict(actor_critic_weighs['state_dict_3'])
        agent_3.optimizer.load_state_dict(actor_critic_weighs['optimizer_state_dict_3'])


    logger = Logger(args.num_processes, envs_dic[0].observation_space.shape, envs_dic[0].observation_space.shape, actor_critic_0.recurrent_hidden_state_size, device=device)

    obs_0 = envs_dic[0].reset()
    obs_1 = envs_dic[1].reset()
    obs_2 = envs_dic[2].reset()
    obs_3 = envs_dic[3].reset()

    rollouts_0.obs[0].copy_(obs_0)
    rollouts_1.obs[0].copy_(obs_1)
    rollouts_2.obs[0].copy_(obs_2)
    rollouts_3.obs[0].copy_(obs_3)


    obs_train = eval_envs_dic['train_eval'].reset()
    logger.obs['train_eval'].copy_(obs_train)
    logger.obs_sum['train_eval'].copy_(obs_train)

    obs_test = eval_envs_dic['test_eval'].reset()
    logger.obs['test_eval'].copy_(obs_test)
    logger.obs_sum['test_eval'].copy_(obs_test)

    fig = plt.figure(figsize=(20, 20))
    columns = 1
    rows = 1
    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(rollouts_0.obs[0][i].transpose(0,2))
        plt.savefig(logdir + '/fig.png')

    seeds = torch.zeros(int(args.num_processes/args.num_c), 1)
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes


    for j in range(args.continue_from_epoch, args.continue_from_epoch+num_updates):


        # Policy rollouts
        actor_critic_0.eval()
        actor_critic_1.eval()
        actor_critic_2.eval()
        actor_critic_3.eval()
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value_0, action_0, action_log_prob_0, _, recurrent_hidden_states_0 = actor_critic_0.act(
                    rollouts_0.obs[step].to(device), rollouts_0.recurrent_hidden_states[step].to(device),
                    rollouts_0.masks[step].to(device))

                value_1, action_1, action_log_prob_1, _, recurrent_hidden_states_1 = actor_critic_1.act(
                    rollouts_1.obs[step].to(device), rollouts_1.recurrent_hidden_states[step].to(device),
                    rollouts_1.masks[step].to(device))


                value_2, action_2, action_log_prob_2, _, recurrent_hidden_states_2 = actor_critic_2.act(
                    rollouts_2.obs[step].to(device), rollouts_2.recurrent_hidden_states[step].to(device),
                    rollouts_2.masks[step].to(device))

                value_3, action_3, action_log_prob_3, _, recurrent_hidden_states_3 = actor_critic_3.act(
                    rollouts_3.obs[step].to(device), rollouts_3.recurrent_hidden_states[step].to(device),
                    rollouts_3.masks[step].to(device))

            # Observe reward and next obs
            obs_0, reward_0, done_0, infos_0 = envs_dic[0].step(action_0.squeeze().cpu().numpy())
            obs_1, reward_1, done_1, infos_1 = envs_dic[1].step(action_1.squeeze().cpu().numpy())
            obs_2, reward_2, done_2, infos_2 = envs_dic[2].step(action_2.squeeze().cpu().numpy())
            obs_3, reward_3, done_3, infos_3 = envs_dic[3].step(action_3.squeeze().cpu().numpy())

            for i, info in enumerate(infos_0):
                seeds[i] = info["level_seed"]

            masks_0 = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done_0])
            bad_masks_0 = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos_0])
            masks_1 = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done_1])
            bad_masks_1 = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos_1])
            masks_2 = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done_2])
            bad_masks_2 = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos_2])
            masks_3 = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done_3])
            bad_masks_3 = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos_3])

            rollouts_0.insert(obs_0, recurrent_hidden_states_0, action_0,
                            action_log_prob_0, value_0, torch.from_numpy(reward_0).unsqueeze(1), masks_0, bad_masks_0, seeds, infos_0, obs_0)
            rollouts_1.insert(obs_1, recurrent_hidden_states_1, action_1,
                            action_log_prob_1, value_1, torch.from_numpy(reward_1).unsqueeze(1), masks_1, bad_masks_1, seeds, infos_1, obs_0)
            rollouts_2.insert(obs_2, recurrent_hidden_states_2, action_2,
                            action_log_prob_2, value_2, torch.from_numpy(reward_2).unsqueeze(1), masks_2, bad_masks_2, seeds, infos_2, obs_0)
            rollouts_3.insert(obs_3, recurrent_hidden_states_3, action_3,
                            action_log_prob_3, value_3, torch.from_numpy(reward_3).unsqueeze(1), masks_3, bad_masks_3, seeds, infos_3, obs_0)

        with torch.no_grad():
            next_value_0 = actor_critic_0.get_value(
                rollouts_0.obs[-1].to(device), rollouts_0.recurrent_hidden_states[-1].to(device),
                rollouts_0.masks[-1].to(device),).detach()

            next_value_1 = actor_critic_1.get_value(
                rollouts_1.obs[-1].to(device), rollouts_1.recurrent_hidden_states[-1].to(device),
                rollouts_1.masks[-1].to(device)).detach()

            next_value_2 = actor_critic_2.get_value(
                rollouts_2.obs[-1].to(device), rollouts_2.recurrent_hidden_states[-1].to(device),
                rollouts_2.masks[-1].to(device)).detach()

            next_value_3 = actor_critic_3.get_value(
                rollouts_3.obs[-1].to(device), rollouts_3.recurrent_hidden_states[-1].to(device),
                rollouts_3.masks[-1].to(device)).detach()

        actor_critic_0.train()
        rollouts_0.compute_returns(next_value_0, use_gae=True, gamma=args.gamma, gae_lambda=args.gae_lambda)
        value_loss_0, action_loss_0, dist_entropy_0, dist_KL_epoch_0 = agent_0.update(rollouts_0)
        rollouts_0.after_update()
        rew_batch_0, done_batch_0 = rollouts_0.fetch_log_data()
        logger.feed_train(rew_batch_0, done_batch_0[1:])

        actor_critic_1.train()
        rollouts_1.compute_returns(next_value_1, use_gae=True, gamma=args.gamma, gae_lambda=args.gae_lambda)

        _, _, _, _ = agent_1.update(rollouts_1)

        rollouts_1.after_update()

        actor_critic_2.train()
        rollouts_2.compute_returns(next_value_2, use_gae=True, gamma=args.gamma, gae_lambda=args.gae_lambda)

        _, _, _, _  = agent_2.update(rollouts_2)

        rollouts_2.after_update()

        actor_critic_3.train()
        rollouts_3.compute_returns(next_value_3, use_gae=True, gamma=args.gamma, gae_lambda=args.gae_lambda)

        _, _, _, _  = agent_3.update(rollouts_3)

        rollouts_3.after_update()

        # Save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0 or j == args.continue_from_epoch + num_updates - 1):
            torch.save({'state_dict_0': actor_critic_0.state_dict(), 'optimizer_state_dict_0': agent_0.optimizer.state_dict(),
                        'state_dict_1': actor_critic_1.state_dict(), 'optimizer_state_dict_1': agent_1.optimizer.state_dict(),
                        'state_dict_2': actor_critic_2.state_dict(), 'optimizer_state_dict_2': agent_2.optimizer.state_dict(),
                        'state_dict_3': actor_critic_3.state_dict(), 'optimizer_state_dict_3': agent_3.optimizer.state_dict(),
                        'step': j}, os.path.join(logdir, args.env_name + "-epoch-{}.pt".format(j)))


        # Print some stats
        if j % args.log_interval == 0:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps

            train_statistics = logger.get_train_statistics()
            print(
                "Updates {}, num timesteps {}, num training episodes {} \n Last 128 training episodes: mean/median reward {:.1f}/{:.1f}, "
                "min/max reward {:.1f}/{:.1f}, dist_entropy {} , value_loss {}, action_loss {}, KL_loss {}, unique seeds {} \n"
                .format(j, total_num_steps,
                        logger.num_episodes, train_statistics['Rewards_mean_episodes'],
                        train_statistics['Rewards_median_episodes'], train_statistics['Rewards_min_episodes'], train_statistics['Rewards_max_episodes'], dist_entropy_0, value_loss_0,
                        action_loss_0, dist_KL_epoch_0, np.unique(rollouts_0.seeds.squeeze().numpy()).size))

        # Evaluate agent on evaluation tasks
        if ((args.eval_interval is not None and j % args.eval_interval == 0) or j == args.continue_from_epoch):
            actor_critic_0.eval()
            actor_critic_1.eval()
            actor_critic_2.eval()
            actor_critic_3.eval()
            eval_test_rew, eval_test_done, eval_test_seeds = evaluate_procgen_LEEP(actor_critic_0, actor_critic_1, actor_critic_2, actor_critic_3,
                                                        eval_envs_dic, 'test_eval', device, args.num_steps, logger, deterministic=False)

            logger.feed_eval(eval_test_rew, eval_test_done)

        # Print some stats
        if j % args.log_interval == 0:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            print('Iter {}, num timesteps {}, num training episodes {}, '
                  'dist_entropy {:.3f}, value_loss {:.3f}, action_loss {:.3f}\n'
                  .format(j, total_num_steps, logger.num_episodes, dist_entropy_0, value_loss_0, action_loss_0))
            episode_statistics = logger.get_episode_statistics()

            print(
                'Last {} training episodes: \n'
                'train mean/median reward {:.1f}/{:.1f},\n'
                'train min/max reward {:.1f}/{:.1f}\n'
                .format(args.num_processes,
                        episode_statistics['Rewards/mean_episodes']['train'], episode_statistics['Rewards/median_episodes']['train'],
                        episode_statistics['Rewards/min_episodes']['train'], episode_statistics['Rewards/max_episodes']['train']))

            print(
                'test mean/median reward {:.1f}/{:.1f},\n'
                'test min/max reward {:.1f}/{:.1f}\n'
                .format(episode_statistics['Rewards/mean_episodes']['test'], episode_statistics['Rewards/median_episodes']['test'],
                        episode_statistics['Rewards/min_episodes']['test'], episode_statistics['Rewards/max_episodes']['test']))

            log = [total_num_steps] + [episode_statistics['Rewards/mean_episodes']['train']] + [episode_statistics['Rewards/min_episodes']['train']] + [episode_statistics['Rewards/max_episodes']['train']]
            log += [episode_statistics['Rewards/mean_episodes']['test']] + [episode_statistics['Rewards/min_episodes']['test']] + [episode_statistics['Rewards/max_episodes']['test']]
            progresslog.loc[len(progresslog)] = log

            with open(logdir + '/progress_{}_seed_{}.csv'.format(args.env_name, args.seed), 'w') as f:
                progresslog.to_csv(f, index=False)


    # Training done. Save and clean up
    for i in range(args.num_c):
        envs_dic[0].close()
    for eval_disp_name in EVAL_ENVS:
        eval_envs_dic[eval_disp_name].close()


if __name__ == "__main__":
    main()
