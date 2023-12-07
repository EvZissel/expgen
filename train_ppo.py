
from PPO_LEEP import algo, utils
from PPO_LEEP.arguments import get_args
from PPO_LEEP.envs import make_ProcgenEnvs
from PPO_LEEP.model import Policy,  ImpalaModel
from PPO_LEEP.storage import RolloutStorage
from evaluation import evaluate_procgen
from PPO_LEEP.procgen_wrappers import *
from PPO_LEEP.logger import Logger
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

    logdir_ = args.env_name + '_seed_' + str(args.seed)

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
                      use_backgrounds=True,
                      restrict_themes=args.restrict_themes,
                      use_monochrome_assets=args.use_monochrome_assets,
                      rand_seed=args.seed,
                      center_agent=args.center_agent,
                      mask_size=args.mask_size,
                      normalize_rew=args.normalize_rew,
                      mask_all=args.mask_all,
                      device=device)

    # Test envs
    eval_envs_dic = {}
    eval_envs_dic['train_eval'] = make_ProcgenEnvs(num_envs=args.num_processes,
                                                      env_name=args.env_name,
                                                      start_level=args.start_level,
                                                      num_levels=args.num_level,
                                                      distribution_mode=args.distribution_mode,
                                                      use_generated_assets=args.use_generated_assets,
                                                      use_backgrounds=True,
                                                      restrict_themes=args.restrict_themes,
                                                      use_monochrome_assets=args.use_monochrome_assets,
                                                      rand_seed=args.seed,
                                                      center_agent=args.center_agent,
                                                      mask_size=args.mask_size,
                                                      normalize_rew= args.normalize_rew,
                                                      mask_all=args.mask_all,
                                                      device=device)

    test_start_level = args.start_level + args.num_level + 1
    eval_envs_dic['test_eval'] = make_ProcgenEnvs(num_envs=args.num_processes,
                                                     env_name=args.env_name,
                                                     start_level=test_start_level,
                                                     num_levels=0,
                                                     distribution_mode=args.distribution_mode,
                                                     use_generated_assets=args.use_generated_assets,
                                                     use_backgrounds=True,
                                                     restrict_themes=args.restrict_themes,
                                                     use_monochrome_assets=args.use_monochrome_assets,
                                                     rand_seed=args.seed,
                                                     center_agent=args.center_agent,
                                                     mask_size=args.mask_size,
                                                     normalize_rew=args.normalize_rew,
                                                     mask_all=args.mask_all,
                                                     device=device)

    eval_envs_dic_nondet = {}
    eval_envs_dic_nondet['test_eval_nondet'] =  make_ProcgenEnvs(num_envs=args.num_processes,
                                                     env_name=args.env_name,
                                                     start_level=test_start_level,
                                                     num_levels=0,
                                                     distribution_mode=args.distribution_mode,
                                                     use_generated_assets=args.use_generated_assets,
                                                     use_backgrounds=True,
                                                     restrict_themes=args.restrict_themes,
                                                     use_monochrome_assets=args.use_monochrome_assets,
                                                     rand_seed=args.seed,
                                                     center_agent=args.center_agent,
                                                     mask_size=args.mask_size,
                                                     normalize_rew=args.normalize_rew,
                                                     mask_all=args.mask_all,
                                                     device=device)

    print('done')

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base=ImpalaModel,
        base_kwargs={'recurrent': False,'hidden_size': args.recurrent_hidden_size,'gray_scale': args.gray_scale})
    actor_critic.to(device)


    if args.algo != 'ppo':
        raise print("only PPO is supported")


    # training agent
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
        attention_policy=False,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay)

    # rollout storage for agent
    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size, args.mask_size, device=device)

    # Load previous model
    if (args.continue_from_epoch > 0) and args.save_dir != "":
        save_path = args.save_dir
        actor_critic_weighs = torch.load(os.path.join(save_path, args.env_name + "-epoch-{}.pt".format(args.continue_from_epoch)), map_location=device)
        actor_critic.load_state_dict(actor_critic_weighs['state_dict'])
        agent.optimizer.load_state_dict(actor_critic_weighs['optimizer_state_dict'])


    # Load previous model
    if (args.saved_epoch > 0) and args.save_dir != "":
        save_path = args.save_dir
        actor_critic_weighs = torch.load(os.path.join(save_path, args.load_env_name + "-epoch-{}.pt".format(args.saved_epoch)), map_location=device)
        actor_critic.load_state_dict(actor_critic_weighs['state_dict'])

    logger = Logger(args.num_processes, envs.observation_space.shape, envs.observation_space.shape, actor_critic.recurrent_hidden_state_size, device=device)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)

    obs_train = eval_envs_dic['train_eval'].reset()
    logger.obs['train_eval'].copy_(obs_train)
    logger.obs_sum['train_eval'].copy_(obs_train)

    obs_test = eval_envs_dic['test_eval'].reset()
    logger.obs['test_eval'].copy_(obs_test)
    logger.obs_sum['test_eval'].copy_(obs_test)

    obs_test_nondet = eval_envs_dic_nondet['test_eval_nondet'].reset()
    logger.obs['test_eval_nondet'].copy_(obs_test_nondet)
    logger.obs_sum['test_eval_nondet'].copy_(obs_test_nondet)

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


    seeds_train = np.zeros((args.num_steps, args.num_processes))
    seeds_test = np.zeros((args.num_steps, args.num_processes))


    for j in range(args.continue_from_epoch, args.continue_from_epoch+num_updates):

        # policy rollouts
        actor_critic.eval()
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, _, recurrent_hidden_states, attn_masks, attn_masks1, attn_masks2, attn_masks3 = actor_critic.act(
                    rollouts.obs[step].to(device), rollouts.recurrent_hidden_states[step].to(device),
                    rollouts.masks[step].to(device), rollouts.attn_masks[step].to(device), rollouts.attn_masks1[step].to(device), rollouts.attn_masks2[step].to(device),
                    rollouts.attn_masks3[step].to(device))

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
                            action_log_prob, value, torch.from_numpy(reward).unsqueeze(1), masks, bad_masks, attn_masks, attn_masks1, attn_masks2, attn_masks3, seeds, infos, obs)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1].to(device), rollouts.recurrent_hidden_states[-1].to(device),
                rollouts.masks[-1].to(device), rollouts.attn_masks[-1].to(device), rollouts.attn_masks1[-1].to(device),
                    rollouts.attn_masks2[-1].to(device), rollouts.attn_masks3[-1].to(device)).detach()

        actor_critic.train()
        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy, _  = agent.update(rollouts)

        rollouts.after_update()

        rew_batch, done_batch = rollouts.fetch_log_data()
        logger.feed_train(rew_batch, done_batch[1:])

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0 or j == args.continue_from_epoch + num_updates - 1):
            torch.save({'state_dict': actor_critic.state_dict(), 'optimizer_state_dict': agent.optimizer.state_dict(),
                        'step': j}, os.path.join(logdir, args.env_name + "-epoch-{}.pt".format(j)))

        # print some stats
        if j % args.log_interval == 0:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps

            train_statistics = logger.get_train_val_statistics()
            print(
                "Updates {}, num timesteps {}, num training episodes {} \n Last 128 training episodes: mean/median reward {:.1f}/{:.1f}, "
                "min/max reward {:.1f}/{:.1f}, dist_entropy {} , value_loss {}, action_loss {}, unique seeds {}\n"
                .format(j, total_num_steps,
                        logger.num_episodes, train_statistics['Rewards_mean_episodes'],
                        train_statistics['Rewards_median_episodes'], train_statistics['Rewards_min_episodes'], train_statistics['Rewards_max_episodes'], dist_entropy, value_loss,
                        action_loss, np.unique(rollouts.seeds.squeeze().numpy()).size))

        # evaluate agent on evaluation tasks
        if ((args.eval_interval is not None and j % args.eval_interval == 0) or j == args.continue_from_epoch):
            actor_critic.eval()
            printout = f'Seed {args.seed} Iter {j} '
            eval_dic_rew = {}
            eval_dic_done = {}
            for eval_disp_name in EVAL_ENVS:
                eval_dic_rew[eval_disp_name], eval_dic_done[eval_disp_name] = evaluate_procgen(actor_critic, eval_envs_dic, eval_disp_name,
                                                  args.num_processes, device, args.num_steps, logger)

            eval_test_nondet_rew, eval_test_nondet_done = evaluate_procgen(actor_critic, eval_envs_dic_nondet, 'test_eval_nondet',
                                                  args.num_processes, device, args.num_steps, logger, attention_features=False, det_masks=False, deterministic=False)

            logger.feed_eval(eval_dic_rew['train_eval'], eval_dic_done['train_eval'],eval_dic_rew['test_eval'], eval_dic_done['test_eval'], seeds_train, seeds_test,
                             eval_dic_rew['train_eval'], eval_dic_rew['test_eval'], eval_test_nondet_rew, eval_test_nondet_done)
            episode_statistics = logger.get_episode_statistics()
            print(printout)
            print(episode_statistics)


    # training done. Save and clean up
    envs.close()
    for eval_disp_name in EVAL_ENVS:
        eval_envs_dic[eval_disp_name].close()


if __name__ == "__main__":
    main()
