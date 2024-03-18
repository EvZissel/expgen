import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from collections import deque
import numpy as np


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, obs_shape_full, action_space, recurrent_hidden_state_size, device="cpu"):
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.obs_ds = torch.zeros(num_steps + 1, num_processes, *(obs_shape[0],int(obs_shape[1]/3),int(obs_shape[2]/3)))
        self.obs_ds_sum = torch.zeros(num_steps + 1, num_processes, *(obs_shape[0],int(obs_shape[1]/3),int(obs_shape[2]/3)))
        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1, num_processes, recurrent_hidden_state_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.seeds = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)
        self.info_batch = deque(maxlen=num_steps)

        self.num_steps = num_steps
        self.step = 0
        self.num_processes = num_processes
        self.device = device
        self.obs_sum = torch.zeros(num_processes, *obs_shape_full)
        self.obs0 = torch.zeros(num_processes, *obs_shape_full)
        self.obs_full = torch.zeros(num_processes, *obs_shape_full)
        self.step_env = torch.ones(num_processes, 1)

    def to(self, device):
        self.obs = self.obs.to(device)
        self.obs_ds = self.obs_ds.to(device)
        self.obs_ds_sum = self.obs_ds_sum.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.seeds = self.seeds.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)
        self.obs_sum = self.obs_sum.to(device)
        self.obs0 = self.obs0.to(device)
        self.obs_full = self.obs_full.to(device)
        self.step_env = self.step_env.to(device)

    def insert(self, obs, recurrent_hidden_states, actions, action_log_probs, value_preds, rewards, masks, bad_masks,seeds, info, obs_full):
        self.obs[self.step + 1].copy_(obs)
        self.recurrent_hidden_states[self.step +
                                     1].copy_(recurrent_hidden_states)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.seeds[self.step].copy_(seeds)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)
        self.info_batch.append(info)
        self.obs_sum += 1 * ((obs_full.cpu() - self.obs_full).abs() > 1e-5) #we sum diffrences
        self.obs_full.copy_(obs_full)
        self.step_env += 1

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.obs_ds[0].copy_(self.obs_ds[-1])
        self.obs_ds_sum[0].copy_(self.obs_ds_sum[-1])
        self.obs_sum.copy_(torch.zeros_like(self.obs_full))
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

    def compute_returns(self,
                        next_value,
                        use_gae,
                        gamma,
                        gae_lambda):
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + gamma * self.value_preds[
                    step + 1] * self.masks[step +
                                           1] - self.value_preds[step]
                gae = delta + gamma * gae_lambda * self.masks[step +
                                                              1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1] * \
                                     gamma * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self,
                               advantages,
                               num_mini_batch=None,
                               mini_batch_size=None):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)
        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(
                -1, self.recurrent_hidden_states.size(-1))[indices]
            actions_batch = self.actions.view(-1,
                                              self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1,
                                                                    1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ

    def single_process_feed_forward_generator(self,
                               advantages,
                               process=0,
                               num_mini_batch=None,
                               mini_batch_size=None):
        # Each process corresponds to a different task
        num_steps, num_processes = self.rewards.size()[0:2]
        assert process < num_processes
        batch_size = num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)
        for indices in sampler:
            obs_batch = self.obs[:-1, process, :].view(-1, *self.obs.size()[2:])[indices]
            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1, process, :].view(
                -1, self.recurrent_hidden_states.size(-1))[indices]
            actions_batch = self.actions[:, process, :].view(-1,
                                                             self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1, process, :].view(-1, 1)[indices]
            return_batch = self.returns[:-1, process, :].view(-1, 1)[indices]
            masks_batch = self.masks[:-1, process, :].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs[:, process, :].view(-1,
                                                                    1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[:, process, :].view(-1, 1)[indices]

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ

    def single_process_recurrent_generator(self, advantages, num_mini_batch, process=0):
        num_processes = self.rewards.size(1)
        assert process < num_processes
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        start_ind = process
        obs_batch = []
        recurrent_hidden_states_batch = []
        actions_batch = []
        value_preds_batch = []
        return_batch = []
        masks_batch = []
        old_action_log_probs_batch = []
        adv_targ = []
        for offset in range(num_envs_per_batch):
            ind = perm[start_ind + offset]
            obs_batch.append(self.obs[:-1, ind])
            recurrent_hidden_states_batch.append(
                self.recurrent_hidden_states[0:1, ind])
            actions_batch.append(self.actions[:, ind])
            value_preds_batch.append(self.value_preds[:-1, ind])
            return_batch.append(self.returns[:-1, ind])
            masks_batch.append(self.masks[:-1, ind])
            old_action_log_probs_batch.append(
                self.action_log_probs[:, ind])
            adv_targ.append(advantages[:, ind])

        T, N = self.num_steps, num_envs_per_batch
        # These are all tensors of size (T, N, -1)
        obs_batch = torch.stack(obs_batch, 1)
        actions_batch = torch.stack(actions_batch, 1)
        value_preds_batch = torch.stack(value_preds_batch, 1)
        return_batch = torch.stack(return_batch, 1)
        masks_batch = torch.stack(masks_batch, 1)
        old_action_log_probs_batch = torch.stack(
            old_action_log_probs_batch, 1)
        adv_targ = torch.stack(adv_targ, 1)

        # States is just a (N, -1) tensor
        recurrent_hidden_states_batch = torch.stack(
            recurrent_hidden_states_batch, 1).view(N, -1)

        # Flatten the (T, N, ...) tensors to (T * N, ...)
        obs_batch = _flatten_helper(T, N, obs_batch)
        actions_batch = _flatten_helper(T, N, actions_batch)
        value_preds_batch = _flatten_helper(T, N, value_preds_batch)
        return_batch = _flatten_helper(T, N, return_batch)
        masks_batch = _flatten_helper(T, N, masks_batch)
        old_action_log_probs_batch = _flatten_helper(T, N, \
                old_action_log_probs_batch)
        adv_targ = _flatten_helper(T, N, adv_targ)

        yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
            value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            obs_batch = []
            recurrent_hidden_states_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                obs_batch.append(self.obs[:-1, ind])
                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(
                    self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            T, N = self.num_steps, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            obs_batch = torch.stack(obs_batch, 1)
            actions_batch = torch.stack(actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch, 1).view(N, -1).to(self.device)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            obs_batch = _flatten_helper(T, N, obs_batch).to(self.device)
            actions_batch = _flatten_helper(T, N, actions_batch).to(self.device)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch).to(self.device)
            return_batch = _flatten_helper(T, N, return_batch).to(self.device)
            masks_batch = _flatten_helper(T, N, masks_batch).to(self.device)
            old_action_log_probs_batch = _flatten_helper(T, N, \
                    old_action_log_probs_batch).to(self.device)
            adv_targ = _flatten_helper(T, N, adv_targ).to(self.device)

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ

    def fetch_log_data(self):
        if 'env_reward' in self.info_batch[0][0]:
            rew_batch = []
            for step in range(self.num_steps):
                infos = self.info_batch[step]
                rew_batch.append([info['env_reward'] for info in infos])
            rew_batch = np.array(rew_batch)
        else:
            rew_batch = np.squeeze(self.rewards.numpy())
        if 'env_done' in self.info_batch[0][0]:
            done_batch = []
            for step in range(self.num_steps):
                infos = self.info_batch[step]
                done_batch.append([info['env_done'] for info in infos])
            done_batch = np.array(done_batch)
        else:
            done_batch = np.squeeze(1 - self.masks.numpy())
        return rew_batch, done_batch