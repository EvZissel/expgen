import pathlib

from PPO_maxEnt_LEEP import algo, utils
from PPO_maxEnt_LEEP.arguments import get_args
from PPO_maxEnt_LEEP.envs import make_ProcgenEnvs
from PPO_maxEnt_LEEP.model import Policy,  ImpalaModel
from PPO_maxEnt_LEEP.storage import RolloutStorage
from evaluation import evaluate_procgen
from PPO_maxEnt_LEEP.procgen_wrappers import *
from PPO_maxEnt_LEEP.logger import Logger
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

def main():
    args = get_args()
    import random; random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    logdir_ = args.env_name + '_ppo' + '_seed_' + str(args.seed)
    if args.mask_all:
        logdir_ += '_mask_all'

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

    progresslog = pd.DataFrame(columns=['timesteps', 'train mean', 'train min', 'train max', 'test mean', 'test min', 'test max'])
    torch.set_num_threads(1)
    device = torch.device("cuda:{}".format(args.gpu_device) if args.cuda else "cpu")

    print('making envs...')
    # Training envs
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

    # Test envs
    # Test environments are sampled from the full distribution of levels
    test_start_level = args.start_level + args.num_level + 1
    test_env = make_ProcgenEnvs(num_envs=args.num_processes,
                                                     env_name=args.env_name,
                                                     start_level=test_start_level,
                                                     num_levels=0,
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

    print('complete making envs...')

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base=ImpalaModel,
        base_kwargs={'recurrent': args.recurrent_policy,'hidden_size': args.recurrent_hidden_size,'gray_scale': args.gray_scale})
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
        save_path = pathlib.Path(args.save_dir, args.env + '_ppo_seed_' + args.seed)
        actor_critic_weighs = torch.load(os.path.join(save_path, args.env_name + "-epoch-{}.pt".format(args.continue_from_epoch)), map_location=device)
        actor_critic.load_state_dict(actor_critic_weighs['state_dict'])
        agent.optimizer.load_state_dict(actor_critic_weighs['optimizer_state_dict'])

    logger = Logger(args.num_processes, envs.observation_space.shape, envs.observation_space.shape, actor_critic.recurrent_hidden_state_size, device=device)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)

    obs_test = test_env.reset()
    logger.obs['test_eval'].copy_(obs_test)
    logger.obs_sum['test_eval'].copy_(obs_test)

    fig = plt.figure(figsize=(20, 20))
    columns = 5
    rows = 5
    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(rollouts.obs[0][i].transpose(0,2))
        plt.savefig(logdir + '/fig.png')

    seeds = torch.zeros(args.num_processes, 1)
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes


    for j in range(args.continue_from_epoch, args.continue_from_epoch+num_updates):

        # Policy rollouts
        actor_critic.eval()
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, _, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step].to(device), rollouts.recurrent_hidden_states[step].to(device), rollouts.masks[step].to(device))

            # Observe reward and next obs
            obs, reward, done, infos = envs.step(action.squeeze().cpu().numpy())

            for i, info in enumerate(infos):
                seeds[i] = info["level_seed"]

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, torch.from_numpy(reward).unsqueeze(1), masks, bad_masks, seeds, infos, obs)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1].to(device), rollouts.recurrent_hidden_states[-1].to(device),
                rollouts.masks[-1].to(device)).detach()

        actor_critic.train()
        rollouts.compute_returns(next_value, use_gae=True, gamma=args.gamma, gae_lambda=args.gae_lambda)

        value_loss, action_loss, dist_entropy, _  = agent.update(rollouts)

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
            eval_test_rew, eval_test_done = evaluate_procgen(actor_critic, test_env, 'test_eval', device, args.num_steps, logger, deterministic=False)

            logger.feed_eval(eval_test_rew, eval_test_done)

        # Print some stats
        if j % args.log_interval == 0:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            print('Iter {}, num timesteps {}, num training episodes {}, '
                  'dist_entropy {:.3f}, value_loss {:.3f}, action_loss {:.3f}\n'
                  .format(j, total_num_steps, logger.num_episodes, dist_entropy, value_loss, action_loss))
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

    # Training done. Close and clean up
    envs.close()
    test_env.close()


if __name__ == "__main__":
    main()
