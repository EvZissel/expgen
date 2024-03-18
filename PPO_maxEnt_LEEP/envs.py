
from gym.spaces.box import Box
from procgen import ProcgenEnv
from PPO_maxEnt_LEEP.procgen_wrappers import *

try:
    import dmc2gym
except ImportError:
    pass

try:
    import roboschool
except ImportError:
    pass

try:
    import pybullet_envs
except ImportError:
    pass


def make_ProcgenEnvs(num_envs,
                     env_name,
                     start_level,
                     num_levels,
                     distribution_mode,
                     use_generated_assets,
                     use_backgrounds,
                     restrict_themes,
                     use_monochrome_assets,
                     rand_seed,
                     center_agent=True,
                     use_sequential_levels=False,
                     mask_size=0,
                     normalize_rew=False,
                     mask_all=False,
                     device='cpu',
                     render_mode=None):

    envs = ProcgenEnv(num_envs=num_envs,
                      env_name=env_name,
                      start_level=start_level,
                      num_levels=num_levels,
                      distribution_mode=distribution_mode,
                      use_generated_assets=use_generated_assets,
                      use_backgrounds=use_backgrounds,
                      restrict_themes=restrict_themes,
                      use_monochrome_assets=use_monochrome_assets,
                      rand_seed=rand_seed,
                      center_agent=center_agent,
                      use_sequential_levels=use_sequential_levels,
                      render_mode=render_mode)

    envs = VecExtractDictObs(envs, "rgb")
    if normalize_rew:
        envs = VecNormalize(envs, ob=False)  # normalizing returns, but not the img frames.
    envs = TransposeFrame(envs)
    # envs = MaskFloatFrame(envs,l=mask_size)
    envs = VecPyTorch(envs, device)
    if mask_size > 0:
        envs = MaskFrame(envs,l=mask_size, device=device)
    if mask_all:
        envs = MaskAllFrame(envs)
    envs = ScaledFloatFrame(envs)

    return envs


# Checks whether done was caused my timit limits or not
class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# Can be used to test recurrent policies for Reacher-v2
class MaskGoal(gym.ObservationWrapper):
    def observation(self, observation):
        if self.env._elapsed_steps > 0:
            observation[-2:] = 0
        return observation


class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)


class TransposeImage(TransposeObs):
    def __init__(self, env=None, op=[2, 0, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, "Error: Operation, " + str(op) + ", must be dim3"
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0], [
                obs_shape[self.op[0]], obs_shape[self.op[1]],
                obs_shape[self.op[2]]
            ],
            dtype=self.observation_space.dtype)

    def observation(self, ob):
        return ob.transpose(self.op[0], self.op[1], self.op[2])


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs


    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        # reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info



# Derived from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device=None):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # Wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        if device is None:
            device = torch.device('cpu')
        self.stacked_obs = torch.zeros((venv.num_envs, ) +
                                       low.shape).to(device)

        observation_space = gym.spaces.Box(low=low,
                                           high=high,
                                           dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stacked_obs[:, :-self.shape_dim0] = \
            self.stacked_obs[:, self.shape_dim0:].clone()
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        if torch.backends.cudnn.deterministic:
            self.stacked_obs = torch.zeros(self.stacked_obs.shape)
        else:
            self.stacked_obs.zero_()
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs

    def close(self):
        self.venv.close()
