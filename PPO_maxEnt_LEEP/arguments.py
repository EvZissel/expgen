import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--lr', type=float, default=5e-4, help='learning rate (default: 5e-4)')
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='Adam optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.999,
        help='discount factor for rewards (default: 0.999)')
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.01,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--KL-coef',
        type=float,
        default=0.01,
        help='KL term coefficient (default: 0.01)')
    parser.add_argument(
        '--epsilon_RPO',
        type=float,
        default=0.0,
        help='epsilon RPO coefficient (default: 0.0)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument(
        '--start-level', type=int, default=0, help='start level (default: 0)')
    parser.add_argument(
        '--num-level', type=int, default=200, help='num level (default: 200)')
    parser.add_argument(
        '--num-test-level', type=int, default=200, help='num test level (default: 200)')
    parser.add_argument(
        '--distribution-mode', type=str, default='easy', help='distribution mode for procgen environment (default: easy)')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--num-processes',
        type=int,
        default=32,
        help='how many training CPU processes to use (default: 32)')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=512,
        help='number of forward steps in ppo (default: 5)')
    parser.add_argument(
        '--ppo-epoch',
        type=int,
        default=3,
        help='number of ppo epochs (default: 3)')
    parser.add_argument(
        '--num-mini-batch',
        type=int,
        default=8,
        help='number of batches for ppo (default: 8)')
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.2,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1525,
        help='save interval, one save per n updates (default: 1525)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=10,
        help='eval interval, one eval per n updates (default: 10)')
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=25e6,
        help='number of environment steps to train (default: 25e6)')
    parser.add_argument(
        '--env-name',
        default='maze',
        help='environment to train on (default: maze)')
    parser.add_argument(
        '--log-dir',
        default='./logs',
        help='directory to save agent logs (default: ./logs)')
    parser.add_argument(
        '--save-dir',
        default="",
        help='directory to save agent logs (default: "")')
    parser.add_argument(
        '--save-dir_maxEnt',
        default="",
        help='directory to save agent logs (default: "")')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training (default: False)')
    parser.add_argument(
        '--gpu_device',
        type=int,
        default = int(0),
        required=False,
        help = 'visible device in CUDA (default: 0)')
    parser.add_argument(
        '--use-proper-time-limits',
        action='store_true',
        default=False,
        help='compute returns taking into account time limits (default: False)')
    parser.add_argument(
        '--recurrent-policy',
        action='store_true',
        default=False,
        help='use a recurrent policy (default: False)')
    parser.add_argument(
        '--obs_recurrent',
        action='store_true',
        default=False,
        help='use a recurrent policy and observations input (default: False)')
    parser.add_argument(
        '--recurrent-hidden-size',
        type=int,
        default=int(256),
        required=False,
        help='GRU hidden layer size (default: 256)')
    parser.add_argument(
        '--continue_from_epoch',
        type=int,
        default=0,
        help='load previous training (from model save dir) and continue training from epoch (default: 0)')
    parser.add_argument(
        '--saved_epoch',
        type=int,
        default=0,
        help='load previous training (from model save dir)')
    parser.add_argument(
        '--saved_epoch_maxEnt',
        type=int,
        default=0,
        help='load previous training (from model save dir)')
    parser.add_argument(
        '--task_steps',
        type=int,
        default=512,
        help='number of steps in each task (default: 512)')
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.0,
        help='weight decay in Adam (default: 0.0)')
    parser.add_argument(
        '--no_normalize',
        action='store_true',
        default=False,
        help='no normalize inputs (default: False)')
    parser.add_argument(
        '--normalize_rew',
        action='store_true',
        default=False,
        help='normalize reword (default: False)')
    parser.add_argument(
        '--mask_size',
        type=int,
        default=0,
        help='constant mask size (default: 0)')
    parser.add_argument(
        '--mask_all',
        action='store_true',
        default=False,
        help='mask all frame (default: False)')
    parser.add_argument(
        '--num_c',
        type=int,
        default=4,
        help='number of ensemble environments (default: 4)')
    parser.add_argument(
        '--KLdiv_loss',
        action='store_true',
        default=False,
        help='use the KLdiv loss between the maxEnt policy and the extrinsic reward policy (default: False)')
    parser.add_argument(
        '--use_generated_assets',
        action='store_true',
        default=False,
        help='use_generated_assets = True for maze (default: False)')
    parser.add_argument(
        '--use_backgrounds',
        action='store_true',
        default=False,
        help='use_generated_assets = False for maze (default: False)')
    parser.add_argument(
        '--restrict_themes',
        action='store_true',
        default=False,
        help='use_generated_assets = True for maze (default: False)')
    parser.add_argument(
        '--use_monochrome_assets',
        action='store_true',
        default=False,
        help='use_generated_assets = True for maze (default: False)')
    parser.add_argument(
        '--num_buffer',
        type=int,
        default=500,
        help='number of images to evaluate k-NN (default: 500, maximum: num-steps)')
    parser.add_argument(
        '--neighbor_size',
        type=int,
        default=3,
        help='number of k in k-NN (default: 3)')
    parser.add_argument(
        '--p_norm',
        type=int,
        default=2,
        help='the norm of the k-NN distance. Supports L0 or L2 (default: 2)')
    parser.add_argument(
        '--num_detEnt',
        type=int,
        default=0,
        help='number of deterministic maxEnt steps for ensemble (default: 0)')
    parser.add_argument(
        '--rand_act',
        action='store_true',
        default=False,
        help='if maxEnt step or random step (default: False)')
    parser.add_argument(
        '--gray_scale',
        action='store_true',
        default=False,
        help='if learns form gray scale image (default: False)')
    parser.add_argument(
        '--reset_cont',
        type=int,
        default=1000,
        help='reset int reward count (default: 1000 max env steps)')
    parser.add_argument(
        '--center_agent',
        action='store_true',
        default=False,
        help='if agent is centered (default: False)')
    parser.add_argument(
        '--kernel_size',
        type=int,
        default=3,
        help='average pool kernel size (default: 3)')
    parser.add_argument(
        '--stride',
        type=int,
        default=3,
        help='average pool stride size (default: 3)')
    parser.add_argument(
        '--num_ensemble',
        type=int,
        default=10,
        help='number of ensemble networks (default: 10)')
    parser.add_argument(
        '--num_agree',
        type=int,
        default=6,
        help='number of networks that should agree (default: 6)')
    parser.add_argument(
        '--num_agent',
        type=int,
        default=0,
        help='the main agent number for the ensemble (default: 0)')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args
