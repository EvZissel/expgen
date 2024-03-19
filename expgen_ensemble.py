
from PPO_maxEnt_LEEP import utils
from PPO_maxEnt_LEEP.arguments import get_args
from PPO_maxEnt_LEEP.envs import make_ProcgenEnvs
from PPO_maxEnt_LEEP.model import Policy,ImpalaModel
from evaluation import evaluate_procgen_ensemble
from PPO_maxEnt_LEEP.procgen_wrappers import *
from PPO_maxEnt_LEEP.logger import Logger
import PPO_maxEnt_LEEP.hyperparams as hps
import pandas as pd
import torch

EVAL_ENVS = ['train_eval','test_eval']

def main():
    args = get_args()
    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    logdir_ = args.env_name + '_ppo' + '_seed_' + str(args.seed)
    logdir_ = logdir_ + '_ensemble'
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

    actor_critic = Policy(
        eval_envs_dic['train_eval'].observation_space.shape,
        eval_envs_dic['train_eval'].action_space,
        base=ImpalaModel,
        base_kwargs={'recurrent': False,'hidden_size': args.recurrent_hidden_size})
    actor_critic.to(device)

    actor_critic1 = Policy(
        eval_envs_dic['train_eval'].observation_space.shape,
        eval_envs_dic['train_eval'].action_space,
        base=ImpalaModel,
        base_kwargs={'recurrent': False,'hidden_size': args.recurrent_hidden_size})
    actor_critic1.to(device)

    actor_critic2 = Policy(
        eval_envs_dic['train_eval'].observation_space.shape,
        eval_envs_dic['train_eval'].action_space,
        base=ImpalaModel,
        base_kwargs={'recurrent': False,'hidden_size': args.recurrent_hidden_size})
    actor_critic2.to(device)

    actor_critic3 = Policy(
        eval_envs_dic['train_eval'].observation_space.shape,
        eval_envs_dic['train_eval'].action_space,
        base=ImpalaModel,
        base_kwargs={'recurrent': False,'hidden_size': args.recurrent_hidden_size})
    actor_critic3.to(device)

    actor_critic4 = Policy(
        eval_envs_dic['train_eval'].observation_space.shape,
        eval_envs_dic['train_eval'].action_space,
        base=ImpalaModel,
        base_kwargs={'recurrent': False, 'hidden_size': args.recurrent_hidden_size})
    actor_critic4.to(device)

    actor_critic5 = Policy(
        eval_envs_dic['train_eval'].observation_space.shape,
        eval_envs_dic['train_eval'].action_space,
        base=ImpalaModel,
        base_kwargs={'recurrent': False, 'hidden_size': args.recurrent_hidden_size})
    actor_critic5.to(device)

    actor_critic6 = Policy(
        eval_envs_dic['train_eval'].observation_space.shape,
        eval_envs_dic['train_eval'].action_space,
        base=ImpalaModel,
        base_kwargs={'recurrent': False, 'hidden_size': args.recurrent_hidden_size})
    actor_critic6.to(device)

    actor_critic7 = Policy(
        eval_envs_dic['train_eval'].observation_space.shape,
        eval_envs_dic['train_eval'].action_space,
        base=ImpalaModel,
        base_kwargs={'recurrent': False, 'hidden_size': args.recurrent_hidden_size})
    actor_critic7.to(device)

    actor_critic8 = Policy(
        eval_envs_dic['train_eval'].observation_space.shape,
        eval_envs_dic['train_eval'].action_space,
        base=ImpalaModel,
        base_kwargs={'recurrent': False, 'hidden_size': args.recurrent_hidden_size})
    actor_critic8.to(device)

    actor_critic9 = Policy(
        eval_envs_dic['train_eval'].observation_space.shape,
        eval_envs_dic['train_eval'].action_space,
        base=ImpalaModel,
        base_kwargs={'recurrent': False, 'hidden_size': args.recurrent_hidden_size})
    actor_critic9.to(device)

    actor_critic_maxEnt = Policy(
        eval_envs_dic['train_eval'].observation_space.shape,
        eval_envs_dic['train_eval'].action_space,
        base=ImpalaModel,
        base_kwargs={'recurrent': True,'hidden_size': args.recurrent_hidden_size})
    actor_critic_maxEnt.to(device)

    save_path =  args.env_name + '_ppo_seed_0'
    save_path = os.path.join(os.path.expanduser(args.log_dir), save_path)
    actor_critic_weighs = torch.load(os.path.join(save_path, args.env_name + '-epoch-1524.pt'), map_location=device)
    actor_critic.load_state_dict(actor_critic_weighs['state_dict'])

    save_path =  args.env_name + '_ppo_seed_1'
    save_path = os.path.join(os.path.expanduser(args.log_dir), save_path)
    actor_critic_weighs = torch.load(os.path.join(save_path, args.env_name + '-epoch-1524.pt'), map_location=device)
    actor_critic1.load_state_dict(actor_critic_weighs['state_dict'])

    save_path =  args.env_name + '_ppo_seed_2'
    save_path = os.path.join(os.path.expanduser(args.log_dir), save_path)
    actor_critic_weighs = torch.load(os.path.join(save_path, args.env_name + '-epoch-1524.pt'), map_location=device)
    actor_critic2.load_state_dict(actor_critic_weighs['state_dict'])

    save_path =  args.env_name + '_ppo_seed_3'
    save_path = os.path.join(os.path.expanduser(args.log_dir), save_path)
    actor_critic_weighs = torch.load(os.path.join(save_path, args.env_name + '-epoch-1524.pt'), map_location=device)
    actor_critic3.load_state_dict(actor_critic_weighs['state_dict'])

    save_path =  args.env_name + '_ppo_seed_4'
    save_path = os.path.join(os.path.expanduser(args.log_dir), save_path)
    actor_critic_weighs = torch.load(os.path.join(save_path, args.env_name + '-epoch-1524.pt'),map_location=device)
    actor_critic4.load_state_dict(actor_critic_weighs['state_dict'])

    save_path =  args.env_name + '_ppo_seed_5'
    save_path = os.path.join(os.path.expanduser(args.log_dir), save_path)
    actor_critic_weighs = torch.load(os.path.join(save_path, args.env_name + '-epoch-1524.pt'),map_location=device)
    actor_critic5.load_state_dict(actor_critic_weighs['state_dict'])

    save_path =  args.env_name + '_ppo_seed_6'
    save_path = os.path.join(os.path.expanduser(args.log_dir), save_path)
    actor_critic_weighs = torch.load(os.path.join(save_path, args.env_name + '-epoch-1524.pt'),map_location=device)
    actor_critic6.load_state_dict(actor_critic_weighs['state_dict'])

    save_path =  args.env_name + '_ppo_seed_7'
    save_path = os.path.join(os.path.expanduser(args.log_dir), save_path)
    actor_critic_weighs = torch.load(os.path.join(save_path, args.env_name + '-epoch-1524.pt'),map_location=device)
    actor_critic7.load_state_dict(actor_critic_weighs['state_dict'])

    save_path =  args.env_name + '_ppo_seed_8'
    save_path = os.path.join(os.path.expanduser(args.log_dir), save_path)
    actor_critic_weighs = torch.load(os.path.join(save_path, args.env_name + '-epoch-1524.pt'),map_location=device)
    actor_critic8.load_state_dict(actor_critic_weighs['state_dict'])

    save_path =  args.env_name + '_ppo_seed_9'
    save_path = os.path.join(os.path.expanduser(args.log_dir), save_path)
    actor_critic_weighs = torch.load(os.path.join(save_path, args.env_name + '-epoch-1524.pt'),map_location=device)
    actor_critic9.load_state_dict(actor_critic_weighs['state_dict'])

    save_path =  args.env_name + '_ppo_seed_0_maxEnt'
    save_path = os.path.join(os.path.expanduser(args.log_dir), save_path)
    actor_critic_weighs = torch.load(os.path.join(save_path,  args.env_name + '-epoch-6100.pt'), map_location=device)
    actor_critic_maxEnt.load_state_dict(actor_critic_weighs['state_dict'])

    logger = Logger(args.num_processes, eval_envs_dic['train_eval'].observation_space.shape, eval_envs_dic['train_eval'].observation_space.shape, actor_critic_maxEnt.recurrent_hidden_state_size, device=device)

    obs_train = eval_envs_dic['train_eval'].reset()
    logger.obs['train_eval'].copy_(obs_train)
    logger.obs_sum['train_eval'].copy_(obs_train)

    obs_test = eval_envs_dic['test_eval'].reset()
    logger.obs['test_eval'].copy_(obs_test)
    logger.obs_sum['test_eval'].copy_(obs_test)

    num_env_steps = hps.num_env_steps['ensemble']
    num_updates = int(
        num_env_steps) // args.num_steps // args.num_processes

    for j in range(args.continue_from_epoch, args.continue_from_epoch+num_updates):

        actor_critic.eval()
        actor_critic1.eval()
        actor_critic2.eval()
        actor_critic3.eval()
        actor_critic4.eval()
        actor_critic5.eval()
        actor_critic6.eval()
        actor_critic7.eval()
        actor_critic8.eval()
        actor_critic9.eval()
        actor_critic_maxEnt.eval()
        maze_miner = False
        if (args.env_name == "maze" or args.env_name == "miner"):
            maze_miner = True

        eval_dic_rew = {}
        eval_dic_done = {}
        num_agree = hps.num_agree[args.env_name]
        for eval_disp_name in EVAL_ENVS:
            eval_dic_rew[eval_disp_name], eval_dic_done[eval_disp_name] = evaluate_procgen_ensemble(actor_critic, actor_critic1, actor_critic2, actor_critic3, actor_critic4, actor_critic5, actor_critic6, actor_critic7, actor_critic8, actor_critic9,
                                                                                                    actor_critic_maxEnt, eval_envs_dic, eval_disp_name,
                                                                                                    args.num_processes, device, args.num_steps, logger, deterministic=False, num_detEnt=args.num_detEnt, rand_act=args.rand_act,
                                                                                                    num_ensemble=args.num_ensemble, num_agree=num_agree, maze_miner=maze_miner, num_agent=args.num_agent)


        logger.feed_train(eval_dic_rew['train_eval'], eval_dic_done['train_eval'])
        logger.feed_eval( eval_dic_rew['test_eval'], eval_dic_done['test_eval'])

        # Print some stats
        if j % args.log_interval == 0:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            print('Iter {}, num timesteps {}'
                  .format(j, total_num_steps))
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

    for eval_disp_name in EVAL_ENVS:
        eval_envs_dic[eval_disp_name].close()


if __name__ == "__main__":
    main()
