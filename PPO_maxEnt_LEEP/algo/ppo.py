import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PPO():
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 KLdiv_coeff=0.0,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 num_tasks=0,
                 weight_decay=0.0,
                 KLdiv_loss=False):

        self.actor_critic = actor_critic
        self.num_tasks = num_tasks
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.KLdiv_loss = KLdiv_loss
        self.KLdiv_coeff = KLdiv_coeff


        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.train_parameters = []
        for name, p in actor_critic.named_parameters():
            self.train_parameters.append(p)

        self.optimizer = optim.Adam(self.train_parameters, lr=lr, eps=eps, weight_decay=weight_decay)

    def update(self, rollouts, maxEntAgent=None):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-8)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        kldiv_loss_epoch = 0
        kl_loss = 0
        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generators = [rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)]

            elif self.num_tasks > 0:
                assert self.num_tasks == rollouts.num_processes

                data_generators = [rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)]
            else:
                data_generators = [rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)]
            for sample in zip(*data_generators):
                task_losses = []
                for task in range(len(sample)):
                    obs_batch, recurrent_hidden_states_batch, actions_batch, \
                       value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                            adv_targ = sample[task]

                    # Reshape to do in a single forward pass for all steps
                    values, action_log_probs, dist_entropy, dist_probs, _ = self.actor_critic.evaluate_actions(
                        obs_batch, recurrent_hidden_states_batch, masks_batch,
                        actions_batch)

                    if self.KLdiv_loss:
                        _, _, _, maxEnt_dist_probs, _  = maxEntAgent.actor_critic.evaluate_actions(
                            obs_batch, recurrent_hidden_states_batch, masks_batch,
                            actions_batch)
                        kl_loss = F.kl_div(dist_probs.log(), maxEnt_dist_probs.log(),reduction='batchmean', log_target=True)

                    dist_entropy = dist_entropy
                    ratio = torch.exp(action_log_probs -
                                      old_action_log_probs_batch)
                    surr1 = ratio * adv_targ
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                        1.0 + self.clip_param) * adv_targ
                    action_loss = (-torch.min(surr1, surr2).mean())


                    if self.use_clipped_value_loss:
                        value_pred_clipped = value_preds_batch + \
                            (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                        value_losses = (values - return_batch).pow(2)
                        value_losses_clipped = (
                            value_pred_clipped - return_batch).pow(2)
                        value_loss = (0.5 * torch.max(value_losses,
                                                     value_losses_clipped).mean())
                    else:
                        value_loss = (0.5 * (return_batch - values).pow(2).mean())
                    task_losses.append(value_loss * self.value_loss_coef + action_loss -
                                       dist_entropy * self.entropy_coef + self.KLdiv_coeff * kl_loss)
                total_loss = torch.stack(task_losses).mean()

                total_loss.backward()


                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                if self.KLdiv_loss:
                    kldiv_loss_epoch += kl_loss.item()

                nn.utils.clip_grad_norm_(self.train_parameters,
                                             self.max_grad_norm)

                self.optimizer.step()
                self.optimizer.zero_grad()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        kldiv_loss_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, kldiv_loss_epoch
