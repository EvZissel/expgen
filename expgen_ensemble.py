
from PPO_LEEP import algo
from PPO_LEEP.arguments import get_args
from PPO_LEEP.envs import make_ProcgenEnvs
from PPO_LEEP.model import Policy,ImpalaModel
from PPO_LEEP.storage import RolloutStorage
from evaluation import evaluate_procgen_ensemble
from PPO_LEEP.procgen_wrappers import *
from PPO_LEEP.logger import Logger
import numpy as np
import torch
from PPO_LEEP.distributions import FixedCategorical

EVAL_ENVS = ['train_eval','test_eval']
EVAL_ENVS_nondet = ['train_eval_nondet','test_eval_nondet']

def main():
    args = get_args()

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


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
                      center_agent=args.center_agent,
                      rand_seed=args.seed,
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
                                                      center_agent=args.center_agent,
                                                      rand_seed=args.seed,
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
                                                     center_agent=args.center_agent,
                                                     rand_seed=args.seed,
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
                                                     center_agent=args.center_agent,
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


    rollouts_maxEnt = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.observation_space.shape, envs.action_space,
                              actor_critic_maxEnt.recurrent_hidden_state_size, args.mask_size, device=device)

    save_path = args.save_dir
    # Load previous model
    if (args.continue_from_epoch > 0) and args.save_dir != "":
        actor_critic_weighs = torch.load(os.path.join(save_path, args.env_name + "-epoch-{}.pt".format(args.continue_from_epoch)), map_location=device)
        actor_critic.load_state_dict(actor_critic_weighs['state_dict'])
        agent.optimizer.load_state_dict(actor_critic_weighs['optimizer_state_dict'])

    # Load previous model
    save_path =  args.env_name + '_seed_0.pt'
    actor_critic_weighs = torch.load(os.path.join(save_path, args.load_env_name + '-epoch-1524.pt'), map_location=device)
    actor_critic.load_state_dict(actor_critic_weighs['state_dict'])

    save_path = args.env_name + '_seed_1.pt'
    actor_critic_weighs = torch.load(os.path.join(save_path, args.env_name + '-epoch-1524.pt'), map_location=device)
    actor_critic1.load_state_dict(actor_critic_weighs['state_dict'])

    save_path = args.env_name + '_seed_2.pt'
    actor_critic_weighs = torch.load(os.path.join(save_path, args.env_name + '-epoch-1524.pt'), map_location=device)
    actor_critic2.load_state_dict(actor_critic_weighs['state_dict'])

    save_path = args.env_name + '_seed_3.pt'
    actor_critic_weighs = torch.load(os.path.join(save_path, args.env_name + '-epoch-1524.pt'), map_location=device)
    actor_critic3.load_state_dict(actor_critic_weighs['state_dict'])

    save_path = args.env_name + '_seed_4.pt'
    actor_critic_weighs = torch.load(os.path.join(save_path, args.env_name + '-epoch-1524.pt'),map_location=device)
    actor_critic4.load_state_dict(actor_critic_weighs['state_dict'])

    save_path = args.env_name + '_seed_5.pt'
    actor_critic_weighs = torch.load(os.path.join(save_path, args.env_name + '-epoch-1524.pt'),map_location=device)
    actor_critic5.load_state_dict(actor_critic_weighs['state_dict'])

    save_path = args.env_name + '_seed_6.pt'
    actor_critic_weighs = torch.load(os.path.join(save_path, args.env_name + '-epoch-1524.pt'),map_location=device)
    actor_critic6.load_state_dict(actor_critic_weighs['state_dict'])

    save_path = args.env_name + '_seed_7.pt'
    actor_critic_weighs = torch.load(os.path.join(save_path, args.env_name + '-epoch-1524.pt'),map_location=device)
    actor_critic7.load_state_dict(actor_critic_weighs['state_dict'])

    save_path = args.env_name + '_seed_8.pt'
    actor_critic_weighs = torch.load(os.path.join(save_path, args.env_name + '-epoch-1524.pt'),map_location=device)
    actor_critic8.load_state_dict(actor_critic_weighs['state_dict'])

    save_path = args.env_name + '_seed_9.pt'
    actor_critic_weighs = torch.load(os.path.join(save_path, args.env_name + '-epoch-1524.pt'),map_location=device)
    actor_critic9.load_state_dict(actor_critic_weighs['state_dict'])

    save_path = args.env_name + '_seed_9_maxEnt.pt'
    actor_critic_weighs = torch.load(os.path.join(save_path,  args.env_name + '-epoch-6102.pt'), map_location=device)
    actor_critic_maxEnt.load_state_dict(actor_critic_weighs['state_dict'])

    logger = Logger(args.num_processes, envs.observation_space.shape, envs.observation_space.shape, actor_critic_maxEnt.recurrent_hidden_state_size, device=device)

    obs = envs.reset()
    rollouts_maxEnt.obs[0].copy_(obs)

    obs_train = eval_envs_dic['train_eval'].reset()
    logger.obs['train_eval'].copy_(obs_train)
    logger.obs_sum['train_eval'].copy_(obs_train)

    obs_test = eval_envs_dic['test_eval'].reset()
    logger.obs['test_eval'].copy_(obs_test)
    logger.obs_sum['test_eval'].copy_(obs_test)

    obs_test_nondet = eval_envs_dic_nondet['test_eval_nondet'].reset()
    logger.obs['test_eval_nondet'].copy_(obs_test_nondet)
    logger.obs_sum['test_eval_nondet'].copy_(obs_test_nondet)


    seeds = torch.zeros(args.num_processes, 1)
    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

    seeds_train = np.zeros((args.num_steps, args.num_processes))
    seeds_test = np.zeros((args.num_steps, args.num_processes))


    m = FixedCategorical(torch.tensor([0.55, 0.25, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025]).repeat(args.num_processes, 1)) # worked for maze #approximrtly Geometric distribution with \alpha = 0.5
    rand_action = FixedCategorical(torch.tensor([ 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 1-14*0.067]).repeat(args.num_processes, 1))
    maxEnt_steps = torch.zeros(args.num_processes,1, device=device)


    for j in range(args.continue_from_epoch, args.continue_from_epoch+num_updates):

        # policy rollouts
        actor_critic.eval()

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action0, action_log_prob, _, recurrent_hidden_states, attn_masks, attn_masks1, attn_masks2, attn_masks3 = actor_critic.act(
                    rollouts_maxEnt.obs[step].to(device), rollouts_maxEnt.recurrent_hidden_states[step].to(device),
                    rollouts_maxEnt.masks[step].to(device), rollouts_maxEnt.attn_masks[step].to(device),
                    rollouts_maxEnt.attn_masks1[step].to(device), rollouts_maxEnt.attn_masks2[step].to(device),
                    rollouts_maxEnt.attn_masks3[step].to(device))

                value, action1, action_log_prob, _, recurrent_hidden_states1, attn_masks, attn_masks1, attn_masks2, attn_masks3 = actor_critic1.act(
                    rollouts_maxEnt.obs[step].to(device), rollouts_maxEnt.recurrent_hidden_states[step].to(device),
                    rollouts_maxEnt.masks[step].to(device), rollouts_maxEnt.attn_masks[step].to(device),
                    rollouts_maxEnt.attn_masks1[step].to(device), rollouts_maxEnt.attn_masks2[step].to(device),
                    rollouts_maxEnt.attn_masks3[step].to(device))

                value, action2, action_log_prob, _, recurrent_hidden_states2, attn_masks, attn_masks1, attn_masks2, attn_masks3 = actor_critic2.act(
                    rollouts_maxEnt.obs[step].to(device), rollouts_maxEnt.recurrent_hidden_states[step].to(device),
                    rollouts_maxEnt.masks[step].to(device), rollouts_maxEnt.attn_masks[step].to(device),
                    rollouts_maxEnt.attn_masks1[step].to(device), rollouts_maxEnt.attn_masks2[step].to(device),
                    rollouts_maxEnt.attn_masks3[step].to(device))

                value, action3, action_log_prob, _, recurrent_hidden_states3, attn_masks, attn_masks1, attn_masks2, attn_masks3 = actor_critic3.act(
                    rollouts_maxEnt.obs[step].to(device), rollouts_maxEnt.recurrent_hidden_states[step].to(device),
                    rollouts_maxEnt.masks[step].to(device), rollouts_maxEnt.attn_masks[step].to(device),
                    rollouts_maxEnt.attn_masks1[step].to(device), rollouts_maxEnt.attn_masks2[step].to(device),
                    rollouts_maxEnt.attn_masks3[step].to(device))

                value, action_maxEnt, action_log_prob, _, recurrent_hidden_states_maxEnt, attn_masks, attn_masks1, attn_masks2, attn_masks3 = actor_critic_maxEnt.act(
                    rollouts_maxEnt.obs[step].to(device), rollouts_maxEnt.recurrent_hidden_states[step].to(device),
                    rollouts_maxEnt.masks[step].to(device), rollouts_maxEnt.attn_masks[step].to(device),
                    rollouts_maxEnt.attn_masks1[step].to(device), rollouts_maxEnt.attn_masks2[step].to(device),
                    rollouts_maxEnt.attn_masks3[step].to(device))

                if args.num_ensemble > 4:
                    value, action4, action_log_prob, _, _, attn_masks, attn_masks1, attn_masks2, attn_masks3 = actor_critic4.act(
                        rollouts_maxEnt.obs[step].to(device), rollouts_maxEnt.recurrent_hidden_states[step].to(device),
                        rollouts_maxEnt.masks[step].to(device), rollouts_maxEnt.attn_masks[step].to(device),
                        rollouts_maxEnt.attn_masks1[step].to(device), rollouts_maxEnt.attn_masks2[step].to(device),
                        rollouts_maxEnt.attn_masks3[step].to(device))

                    value, action5, action_log_prob, _, _, attn_masks, attn_masks1, attn_masks2, attn_masks3 = actor_critic5.act(
                        rollouts_maxEnt.obs[step].to(device), rollouts_maxEnt.recurrent_hidden_states[step].to(device),
                        rollouts_maxEnt.masks[step].to(device), rollouts_maxEnt.attn_masks[step].to(device),
                        rollouts_maxEnt.attn_masks1[step].to(device), rollouts_maxEnt.attn_masks2[step].to(device),
                        rollouts_maxEnt.attn_masks3[step].to(device))

                if args.num_ensemble > 6:
                    value, action6, action_log_prob, _, recurrent_hidden_states2, attn_masks, attn_masks1, attn_masks2, attn_masks3 = actor_critic6.act(
                        rollouts_maxEnt.obs[step].to(device), rollouts_maxEnt.recurrent_hidden_states[step].to(device),
                        rollouts_maxEnt.masks[step].to(device), rollouts_maxEnt.attn_masks[step].to(device),
                        rollouts_maxEnt.attn_masks1[step].to(device), rollouts_maxEnt.attn_masks2[step].to(device),
                        rollouts_maxEnt.attn_masks3[step].to(device))

                    value, action7, action_log_prob, _, recurrent_hidden_states3, attn_masks, attn_masks1, attn_masks2, attn_masks3 = actor_critic7.act(
                        rollouts_maxEnt.obs[step].to(device), rollouts_maxEnt.recurrent_hidden_states[step].to(device),
                        rollouts_maxEnt.masks[step].to(device), rollouts_maxEnt.attn_masks[step].to(device),
                        rollouts_maxEnt.attn_masks1[step].to(device), rollouts_maxEnt.attn_masks2[step].to(device),
                        rollouts_maxEnt.attn_masks3[step].to(device))

                if args.num_ensemble > 8:
                    value, action8, action_log_prob, _, recurrent_hidden_states2, attn_masks, attn_masks1, attn_masks2, attn_masks3 = actor_critic8.act(
                        rollouts_maxEnt.obs[step].to(device), rollouts_maxEnt.recurrent_hidden_states[step].to(device),
                        rollouts_maxEnt.masks[step].to(device), rollouts_maxEnt.attn_masks[step].to(device),
                        rollouts_maxEnt.attn_masks1[step].to(device), rollouts_maxEnt.attn_masks2[step].to(device),
                        rollouts_maxEnt.attn_masks3[step].to(device))

                    value, action9, action_log_prob, _, recurrent_hidden_states3, attn_masks, attn_masks1, attn_masks2, attn_masks3 = actor_critic9.act(
                        rollouts_maxEnt.obs[step].to(device), rollouts_maxEnt.recurrent_hidden_states[step].to(device),
                        rollouts_maxEnt.masks[step].to(device), rollouts_maxEnt.attn_masks[step].to(device),
                        rollouts_maxEnt.attn_masks1[step].to(device), rollouts_maxEnt.attn_masks2[step].to(device),
                        rollouts_maxEnt.attn_masks3[step].to(device))


            actions = []
            actions.append(action0)
            actions.append(action1)
            actions.append(action2)
            actions.append(action3)
            cardinal_left = 1*(action0 == 0)+ 1*(action0 == 1) + 1*(action0 == 2) + 1*(action1 == 0)+1*(action1 == 1) + 1*(action1 == 2) + 1*(action2 == 0)+1*(action2 == 1) + 1*(action2 == 2)\
                            + 1 * (action3 == 0) + 1 * (action3 == 1) + 1 * (action3 == 2)
            cardinal_right  = 1*(action0 == 6)+1*(action0 == 7) + 1*(action0 == 8) + 1*(action1 == 6)+1*(action1 == 7) + 1*(action1 == 8) + 1*(action2 == 6)+1*(action2 == 7) + 1*(action2 == 8)\
                            + 1 * (action3 == 6) + 1 * (action3 == 7) + 1 * (action3 == 8)
            if (args.env_name=="maze" or args.env_name=="miner"):
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

            if args.num_ensemble > 4:
                actions.append(action4)
                actions.append(action5)
                cardinal_left += 1 * (action4 == 0) + 1 * (action4 == 1) + 1 * (action4 == 2) + 1 * (action5 == 0) + 1 * (action5 == 1) + 1 * (action5 == 2)
                cardinal_right += 1 * (action4 == 6) + 1 * (action4 == 7) + 1 * (action4 == 8) + 1 * (action5 == 6) + 1 * (action5 == 7) + 1 * (action5 == 8)
                if (args.env_name == "maze" or args.env_name == "miner"):
                    cardinal_down += 1 * (action4 == 3) + 1 * (action5 == 3)
                    cardinal_up += 1 * (action4 == 5) + 1 * (action5 == 5)
                else:
                    cardinal_down += 1 * (action4 == 3) + 1 * (action5 == 3) + 1 * (action4 == 0) + 1 * (action5 == 0) + 1 * (action4 == 6) + 1 * (action5 == 6)
                    cardinal_up += 1 * (action4 == 5) + 1 * (action5 == 5) + 1 * (action4 == 2) + 1 * (action5 == 2) + 1 * (action4 == 8) + 1 * (action5 == 8)
                    cardinal_fire += 1 * (action4 == 9) + 1 * (action5 == 9)
                    cardinal_else += 1 * (action4 == 4) + 1 * (action4 == 10) + 1 * (action4 == 11) + 1 * (action4 == 12) + 1 * (action4 == 13) + 1 * (action4 == 14) \
                                  + 1 * (action5 == 9) + 1 * (action5 == 10) + 1 * (action5 == 11) + 1 * (action5 == 12) + 1 * (action5 == 13) + 1 * (action5 == 14)

            if args.num_ensemble > 6:
                actions.append(action6)
                actions.append(action7)
                cardinal_left += 1 * (action6 == 0) + 1 * (action6 == 1) + 1 * (action6 == 2) + 1 * (action7 == 0) + 1 * (action7 == 1) + 1 * (action7 == 2)
                cardinal_right += 1 * (action6 == 6) + 1 * (action6 == 7) + 1 * (action6 == 8) + 1 * (action7 == 6) + 1 * (action7 == 7) + 1 * (action7 == 8)
                if (args.env_name == "maze" or args.env_name == "miner"):
                    cardinal_down += 1 * (action6 == 3) + 1 * (action7 == 3)
                    cardinal_up += 1 * (action6 == 5) + 1 * (action7 == 5)
                else:
                    cardinal_down += 1 * (action6 == 3) + 1 * (action7 == 3) + 1 * (action6 == 0) + 1 * (action7 == 0) + 1 * (action6 == 6) + 1 * (action7 == 6)
                    cardinal_up += 1 * (action6 == 5) + 1 * (action7 == 5) + 1 * (action6 == 2) + 1 * (action7 == 2) + 1 * (action6 == 8) + 1 * (action7 == 8)
                    cardinal_fire += 1 * (action6 == 9) + 1 * (action7 == 9)
                    cardinal_else += 1 * (action6 == 4) + 1 * (action6 == 10) + 1 * (action6 == 11) + 1 * (action6 == 12) + 1 * (action6 == 13) + 1 * (action6 == 14) \
                                  + 1 * (action7 == 9) + 1 * (action7 == 10) + 1 * (action7 == 11) + 1 * (action7 == 12) + 1 * (action7 == 13) + 1 * (action7 == 14)

            if args.num_ensemble > 8:
                actions.append(action8)
                actions.append(action9)
                cardinal_left += 1 * (action8 == 0) + 1 * (action8 == 1) + 1 * (action8 == 2) + 1 * (action9 == 0) + 1 * (action9 == 1) + 1 * (action9 == 2)
                cardinal_right += 1 * (action8 == 6) + 1 * (action8 == 7) + 1 * (action8 == 8) + 1 * (action9 == 6) + 1 * (action9 == 7) + 1 * (action9 == 8)
                if (args.env_name == "maze" or args.env_name == "miner"):
                    cardinal_down += 1 * (action8 == 3) + 1 * (action9 == 3)
                    cardinal_up += 1 * (action8 == 5) + 1 * (action9 == 5)
                else:
                    cardinal_down += 1 * (action8 == 3) + 1 * (action9 == 3) + 1 * (action8 == 0) + 1 * (action9 == 0) + 1 * (action8 == 6) + 1 * (action9 == 6)
                    cardinal_up += 1 * (action8 == 5) + 1 * (action9 == 5) + 1 * (action8 == 2) + 1 * (action9 == 2) + 1 * (action8 == 8) + 1 * (action9 == 8)
                    cardinal_fire += 1 * (action8 == 9) + 1 * (action9 == 9)
                    cardinal_else += 1 * (action8 == 4) + 1 * (action8 == 10) + 1 * (action8 == 11) + 1 * (action8 == 12) + 1 * (action8 == 13) + 1 * (action8 == 14) \
                                  + 1 * (action9 == 9) + 1 * (action9 == 10) + 1 * (action9 == 11) + 1 * (action9 == 12) + 1 * (action9 == 13) + 1 * (action9 == 14)

            if (args.env_name == "maze" or args.env_name == "miner"):
                directions = torch.cat((cardinal_up, cardinal_right, cardinal_down, cardinal_left), dim=1)
            else:
                directions = torch.cat((cardinal_up, cardinal_right, cardinal_down, cardinal_left, cardinal_fire, cardinal_else), dim=1)

            action_cardinal_left =  1 * ( actions[args.num_agent] == 0) + 1 * ( actions[args.num_agent] == 1) + 1 * ( actions[args.num_agent] == 2)
            action_cardinal_right =  1 * ( actions[args.num_agent] == 6) + 1 * ( actions[args.num_agent] == 7) + 1 * ( actions[args.num_agent] == 8)
            if (args.env_name == "maze" or args.env_name == "miner"):
                action_cardinal_down = 1 * (actions[args.num_agent] == 3)
                action_cardinal_up = 1 * (actions[args.num_agent] == 5)
                action_directions = torch.cat((action_cardinal_up, action_cardinal_right, action_cardinal_down, action_cardinal_left), dim=1)
            else:
                action_cardinal_down = 1 * (actions[args.num_agent] == 3) + 1 * (actions[args.num_agent] == 0) + 1 * (actions[args.num_agent] == 6)
                action_cardinal_up = 1 * (actions[args.num_agent] == 5) + 1 * (actions[args.num_agent] == 2) + 1 * (actions[args.num_agent] == 8)
                action_cardinal_fire = 1 * (actions[args.num_agent] == 9)
                action_cardinal_else = 1 * (actions[args.num_agent] == 4) + 1 * (actions[args.num_agent] == 10) + 1 * (actions[args.num_agent] == 11) + 1 * (actions[args.num_agent] == 12) + 1 * (actions[args.num_agent] == 13) + 1 * (actions[args.num_agent] == 14)
                action_directions = torch.cat((action_cardinal_up, action_cardinal_right, action_cardinal_down, action_cardinal_left, action_cardinal_fire, action_cardinal_else), dim=1)

            action_cardinal_index = torch.max(action_directions, dim=1)[1]


            is_majority = (directions[torch.arange(32),action_cardinal_index] >= args.num_agree).unsqueeze(1)
            action_NN = actions[args.num_agent]

            maxEnt_steps = maxEnt_steps - 1

            maxEnt_steps_sample = (~is_majority)*(maxEnt_steps<=0)
            maxEnt_steps = (m.sample() + 1).to(device)*maxEnt_steps_sample + maxEnt_steps*(~maxEnt_steps_sample)

            is_action = is_majority*(maxEnt_steps<=0)
            action = action_NN*is_action + action_maxEnt*(~is_action)
            if args.rand_act:
                action = action_NN * is_action + rand_action.sample().to(device) * (~is_action)
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
            rollouts_maxEnt.insert(obs, recurrent_hidden_states_maxEnt, action,
                            action_log_prob, value, torch.from_numpy(reward).unsqueeze(1), masks, bad_masks, attn_masks,
                            attn_masks1, attn_masks2, attn_masks3, seeds, infos, obs)


        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts_maxEnt.obs[-1].to(device), rollouts_maxEnt.recurrent_hidden_states[-1].to(device),
                rollouts_maxEnt.masks[-1].to(device), rollouts_maxEnt.attn_masks[-1].to(device), rollouts_maxEnt.attn_masks1[-1].to(device),
                rollouts_maxEnt.attn_masks2[-1].to(device), rollouts_maxEnt.attn_masks3[-1].to(device)).detach()

        actor_critic.train()
        rollouts_maxEnt.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        rollouts_maxEnt.after_update()

        rew_batch, done_batch = rollouts_maxEnt.fetch_log_data()
        logger.feed_train(rew_batch, done_batch[1:])


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
        for eval_disp_name in EVAL_ENVS:
            eval_dic_rew[eval_disp_name], eval_dic_done[eval_disp_name] = evaluate_procgen_ensemble(actor_critic, actor_critic1, actor_critic2, actor_critic3, actor_critic4, actor_critic5, actor_critic6, actor_critic7, actor_critic8, actor_critic9,
                                                                                                    actor_critic_maxEnt, eval_envs_dic, eval_disp_name,
                                                                                                    args.num_processes, device, args.num_steps, logger, deterministic=True, num_detEnt=args.num_detEnt, rand_act=args.rand_act,
                                                                                                    num_ensemble=args.num_ensemble, num_agree=args.num_agree, maze_miner=maze_miner, num_agent=args.num_agent)


        eval_test_nondet_rew, eval_test_nondet_done = evaluate_procgen_ensemble(actor_critic, actor_critic1, actor_critic2, actor_critic3, actor_critic4, actor_critic5, actor_critic6, actor_critic7, actor_critic8, actor_critic9,
                                                                                actor_critic_maxEnt, eval_envs_dic_nondet, 'test_eval_nondet',
                                                                                args.num_processes, device, args.num_steps, logger, attention_features=False, det_masks=False, deterministic=False, num_detEnt=args.num_detEnt, rand_act=args.rand_act,
                                                                                num_ensemble=args.num_ensemble, num_agree=args.num_agree, maze_miner=maze_miner, num_agent=args.num_agent)

        logger.feed_eval(eval_dic_rew['train_eval'], eval_dic_done['train_eval'], eval_dic_rew['test_eval'], eval_dic_done['test_eval'], seeds_train, seeds_test,
                         eval_dic_rew['train_eval'], eval_dic_rew['test_eval'], eval_test_nondet_rew, eval_test_nondet_done)
        if len(logger.episode_reward_buffer)>0 and len(logger.episode_reward_buffer_test_nondet)>0:
            episode_statistics = logger.get_episode_statistics()
            print("train and test eval")
            print(episode_statistics)


    for eval_disp_name in EVAL_ENVS:
        eval_envs_dic[eval_disp_name].close()


if __name__ == "__main__":
    main()
