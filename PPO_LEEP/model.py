import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import RelaxedBernoulli

from PPO_LEEP.distributions import Bernoulli, Categorical, DiagGaussian
from PPO_LEEP.utils import init


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None, epsilon_RPO=0):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError

        self.base = base(obs_shape[0], **base_kwargs)
        self.epsilon_RPO = epsilon_RPO

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs, epsilon_RPO)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    @property
    def is_attention(self):
        """Size of attention."""
        return self.base.is_attention

    @property
    def attention_size(self):
        """Size of attention."""
        return self.base.attention_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, attn_masks=None, attn_masks1=None, attn_masks2=None, attn_masks3=None, deterministic=False, attention_act=False, reuse_masks=False):
        value, actor_features, rnn_hxs, attn_log_probs, attn_masks, attn_masks1, attn_masks2, attn_masks3 = self.base(inputs, rnn_hxs, masks, attn_masks, attn_masks1, attn_masks2,
                                                                                                                      attn_masks3, reuse_masks=reuse_masks)

        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        # we return the log probs of either the chosen action, or chosen attention mask
        if attention_act:
            action_log_probs = attn_log_probs
        else:
            action_log_probs = dist.log_probs(action)
        dist_probs = dist.probs
        return value, action, action_log_probs, dist_probs, rnn_hxs, attn_masks, attn_masks1, attn_masks2, attn_masks3

    def get_value(self, inputs, rnn_hxs, masks, attn_masks, attn_masks1, attn_masks2, attn_masks3):
        value, _, _, _, _, _, _, _ = self.base(inputs, rnn_hxs, masks, attn_masks, attn_masks1, attn_masks2, attn_masks3)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, attn_masks, attn_masks1, attn_masks2, attn_masks3, action, attention_act=False):
        if attention_act:
            action_log_probs = self.base.attn_log_probs(attn_masks, attn_masks1, attn_masks2, attn_masks3)
            value = torch.tensor(0)
            dist_entropy = torch.tensor(0)

        else:
            value, actor_features, rnn_hxs, attn_log_probs, _, _, _, _ = self.base(inputs, rnn_hxs, masks, attn_masks, attn_masks1, attn_masks2, attn_masks3,
                                                                          reuse_masks=True)
            dist = self.dist(actor_features)
            action_log_probs = dist.log_probs(action)
            dist_entropy = dist.entropy().mean()
            dist_probs = dist.probs

        return value, action_log_probs, dist_entropy, dist_probs,  rnn_hxs


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size, attention=False, attention_size=0):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent
        self._attention = attention
        self._attention_size = attention_size

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def is_attention(self):
        return self._attention

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def attention_size(self):
        if self._attention:
            return self._attention_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class Flatten(nn.Module):
    def forward(self, x):
        # return x.view(x.size(0), -1)
        return x.reshape(x.size(0), -1)

class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels):
        super(ResidualBlock, self).__init__()
        # init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
        #                        constant_(x, 0), nn.init.calculate_gain('relu'))
        init_ = lambda m: init(m, nn.init.xavier_uniform_, lambda x: nn.init.
                               constant_(x, 0))


        self.conv1 = init_(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1))
        self.conv2 = init_(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        out = nn.ReLU()(x)
        out = self.conv1(out)
        out = nn.ReLU()(out)
        out = self.conv2(out)
        return out + x


class ImpalaBlockAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ImpalaBlockAttention, self).__init__()
        init_ = lambda m: init(m, nn.init.xavier_uniform_, lambda x: nn.init.
                               constant_(x, 0))

        self.attention = nn.Parameter(torch.ones(out_channels), requires_grad=True)

        self.conv = init_(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1))
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)


    def forward(self, x, masks, attn_masks=None, reuse_masks=False):
        probs = torch.sigmoid(self.attention)
        probs = torch.distributions.bernoulli.Bernoulli(probs=probs)
        new_attn = probs.sample([x.shape[0]])
        new_attn_masks = attn_masks if reuse_masks else new_attn
        attn_masks = new_attn_masks * (1 - masks) + attn_masks * masks
        attn_log_probs = torch.flatten(probs.log_prob(attn_masks), start_dim=1).sum(dim=1).reshape([x.shape[0], 1])
        m = attn_masks.unsqueeze(2).unsqueeze(3).repeat(1, 1, x.shape[2], x.shape[3])

        x = self.conv(x)
        x = m * x
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)
        x = self.res1(x)
        x = self.res2(x)

        return x, attn_masks, attn_log_probs

    def attn_log_probs(self, attn_masks):
        probs = torch.sigmoid(self.attention)
        probs = torch.distributions.bernoulli.Bernoulli(probs=probs)
        attn_log_probs = torch.flatten(probs.log_prob(attn_masks), start_dim=1).sum(dim=1).reshape(
            [attn_masks.shape[0], 1])

        return attn_log_probs


class ImpalaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ImpalaBlock, self).__init__()
        init_ = lambda m: init(m, nn.init.xavier_uniform_, lambda x: nn.init.
                               constant_(x, 0))

        self.conv = init_(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1))
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)
        x = self.res1(x)
        x = self.res2(x)

        return x

class ImpalaBlock_small(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ImpalaBlock_small, self).__init__()
        init_ = lambda m: init(m, nn.init.xavier_uniform_, lambda x: nn.init.
                               constant_(x, 0))

        self.conv = init_(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1))
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.MaxPool2d(kernel_size=5, stride=4, padding=2)(x)
        x = self.res1(x)
        x = self.res2(x)

        return x


class ImpalaModel(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=256, gray_scale=False):
        super(ImpalaModel, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.xavier_uniform_, lambda x: nn.init.
                               constant_(x, 0))

        init_2 = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=1)

        self.main = nn.Sequential(
            ImpalaBlock(in_channels=num_inputs, out_channels=16),
            ImpalaBlock(in_channels=16, out_channels=32),
            ImpalaBlock(in_channels=32, out_channels=32), nn.ReLU(), Flatten(),
            init_(nn.Linear(in_features=32 * 8 * 8, out_features=hidden_size)),nn.ReLU())

        self.critic_linear = init_2(nn.Linear(hidden_size, 1))
        self.gray = gray_scale

        self.train()

    def forward(self, inputs, rnn_hxs, masks, attn_masks, attn_masks1, attn_masks2, attn_masks3, reuse_masks=False):
        x = inputs
        if self.gray:
            x_gray = x.clone()
            r = x[:,0, :, :]
            g = x[:,1, :, :]
            b = x[:,2, :, :]
            x_gray[:,0, :, :] = 0.2989 * r + 0.5870 * g + 0.1140 * b
            x_gray[:,1, :, :] = 0.2989 * r + 0.5870 * g + 0.1140 * b
            x_gray[:,2, :, :] = 0.2989 * r + 0.5870 * g + 0.1140 * b
            x = x_gray
        x = self.main(x)
        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs, None, attn_masks, attn_masks1, attn_masks2, attn_masks3


class ImpalaModel_finetune(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=256):
        super(ImpalaModel_finetune, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.xavier_uniform_, lambda x: nn.init.
                               constant_(x, 0))

        init_2 = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=1)

        self.main = nn.Sequential(
            ImpalaBlock(in_channels=num_inputs, out_channels=16),
            ImpalaBlock(in_channels=16, out_channels=32),
            ImpalaBlock(in_channels=32, out_channels=32), nn.ReLU(), Flatten(),
            init_(nn.Linear(in_features=32 * 8 * 8, out_features=hidden_size)),nn.ReLU())

        self.critic_linear = init_2(nn.Linear(hidden_size, 1))
        self.fine_tune_adapt = nn.Sequential(init_2(nn.Linear(hidden_size, hidden_size)),nn.ReLU(),init_2(nn.Linear(hidden_size, hidden_size)),nn.ReLU())
        self.train()

    def forward(self, inputs, rnn_hxs, masks, attn_masks, attn_masks1, attn_masks2, attn_masks3, reuse_masks=False):
        x = inputs
        x = self.main(x)
        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        x = self.fine_tune_adapt(x)

        return self.critic_linear(x), x, rnn_hxs, None, attn_masks, attn_masks1, attn_masks2, attn_masks3


class ImpalaModel_small(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=256):
        super(ImpalaModel_small, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.xavier_uniform_, lambda x: nn.init.
                               constant_(x, 0))

        init_2 = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=1)

        self.main = nn.Sequential(
            ImpalaBlock_small(in_channels=num_inputs, out_channels=4), nn.ReLU(), Flatten(),
            init_(nn.Linear(in_features=4 * 16 * 16, out_features=hidden_size)),nn.ReLU())

        self.critic_linear = init_2(nn.Linear(hidden_size, 1))
        self.train()

    def forward(self, inputs, rnn_hxs, masks, attn_masks, attn_masks1, attn_masks2, attn_masks3, reuse_masks=False):
        x = inputs
        x = self.main(x)
        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs, None, attn_masks, attn_masks1, attn_masks2, attn_masks3

class ImpalaHardAttnReinforce(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=256, att_size=None, obs_size=None):
        super(ImpalaHardAttnReinforce, self).__init__(recurrent, hidden_size, hidden_size, attention=True, attention_size=att_size)

        init_ = lambda m: init(m, nn.init.xavier_uniform_, lambda x: nn.init.
                               constant_(x, 0))

        init_2 = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=1)

        self.exp_size = int(obs_size[1]/att_size[0])

        self.input_attention = nn.Parameter(torch.ones(att_size), requires_grad=True)
        self.ones = nn.Parameter(torch.ones(self.exp_size,self.exp_size), requires_grad=False)
        self.brown=nn.Parameter( torch.stack([191*torch.ones(obs_size[1], obs_size[2]),
                                               127*torch.ones(obs_size[1], obs_size[2]),63*torch.ones(obs_size[1], obs_size[2])], dim=0)/255.0, requires_grad=False)

        self.main = nn.Sequential(
            ImpalaBlock(in_channels=num_inputs, out_channels=16),
            ImpalaBlock(in_channels=16, out_channels=32),
            ImpalaBlock(in_channels=32, out_channels=32), nn.ReLU(), Flatten(),
            init_(nn.Linear(in_features=32 * 8 * 8, out_features=256)), nn.ReLU())


        self.critic_linear = init_2(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks, attn_masks, reuse_masks=False):
        x = inputs
        probs = torch.sigmoid(self.input_attention)
        probs = torch.distributions.bernoulli.Bernoulli(probs=probs)
        new_attn = probs.sample([inputs.shape[0]])
        new_attn_masks = attn_masks if reuse_masks else new_attn
        attn_masks = new_attn_masks * (1 - masks.unsqueeze(2)) + attn_masks * masks.unsqueeze(2)
        attn_log_probs = torch.flatten(probs.log_prob(attn_masks), start_dim=1).sum(dim=1).reshape([inputs.shape[0], 1])
        m = torch.kron(attn_masks, self.ones).unsqueeze(1).repeat(1, inputs.shape[1], 1, 1)
        x = m * x + (1-m) * self.brown

        x = self.main(x)
        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs, attn_log_probs, attn_masks

    def attn_log_probs(self, attn_masks):
        probs = torch.sigmoid(self.input_attention)
        probs = torch.distributions.bernoulli.Bernoulli(probs=probs)
        attn_log_probs = torch.flatten(probs.log_prob(attn_masks), start_dim=1).sum(dim=1).reshape([attn_masks.shape[0], 1])

        return attn_log_probs


class ImpalaHardAttnReinforceAllFeatures(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=256, attention_size=32):
        super(ImpalaHardAttnReinforceAllFeatures, self).__init__(recurrent, hidden_size, hidden_size, attention=True, attention_size=attention_size)

        init_ = lambda m: init(m, nn.init.xavier_uniform_, lambda x: nn.init.
                               constant_(x, 0))

        init_2 = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=1)

        self.exp_size = int(hidden_size / attention_size)

        self.linear_attention = nn.Parameter(torch.ones(attention_size), requires_grad=True)
        self.ones = nn.Parameter(torch.ones(self.exp_size), requires_grad=False)

        self.block1 = ImpalaBlockAttention(in_channels=num_inputs, out_channels=16)
        self.block2 = ImpalaBlockAttention(in_channels=16, out_channels=32)
        self.block3 = ImpalaBlockAttention(in_channels=32, out_channels=32)
        self.linear = init_(nn.Linear(in_features=32 * 8 * 8, out_features=256))

        self.critic_linear = init_2(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks, attn_masks, attn_masks1, attn_masks2, attn_masks3, reuse_masks=False):
        x = inputs
        probs = torch.sigmoid(self.linear_attention)
        probs = torch.distributions.bernoulli.Bernoulli(probs=probs)
        new_attn = probs.sample([inputs.shape[0]])
        new_attn_masks = attn_masks if reuse_masks else new_attn
        attn_masks = new_attn_masks * (1 - masks) + attn_masks * masks
        attn_log_probs = torch.flatten(probs.log_prob(attn_masks), start_dim=1).sum(dim=1).reshape([inputs.shape[0], 1])
        m = torch.kron(attn_masks, self.ones)


        x, attn_masks1, attn_log_probs1 = self.block1(x, masks, attn_masks1, reuse_masks=reuse_masks)
        x, attn_masks2, attn_log_probs2 = self.block2(x, masks, attn_masks2, reuse_masks=reuse_masks)
        x, attn_masks3, attn_log_probs3 = self.block3(x, masks, attn_masks3, reuse_masks=reuse_masks)
        x = nn.ReLU()(x)
        x = Flatten()(x)
        x = self.linear(x)
        x = nn.ReLU()(x)
        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        x = m * x

        return self.critic_linear(x), x, rnn_hxs, attn_log_probs+attn_log_probs1+attn_log_probs2+attn_log_probs3, attn_masks, attn_masks1,  attn_masks2, attn_masks3

    def attn_log_probs(self, attn_masks, attn_masks1,  attn_masks2, attn_masks3):
        probs = torch.sigmoid(self.linear_attention)
        probs = torch.distributions.bernoulli.Bernoulli(probs=probs)
        attn_log_probs = torch.flatten(probs.log_prob(attn_masks), start_dim=1).sum(dim=1).reshape(
            [attn_masks.shape[0], 1])


        attn_log_probs1 = self.block1.attn_log_probs(attn_masks1)
        attn_log_probs2 = self.block2.attn_log_probs(attn_masks2)
        attn_log_probs3 = self.block3.attn_log_probs(attn_masks3)

        return attn_log_probs+attn_log_probs1+attn_log_probs2+attn_log_probs3


class ImpalaHardAttnReinforceFeatures(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=256, attention_size=32):
        super(ImpalaHardAttnReinforceFeatures, self).__init__(recurrent, hidden_size, hidden_size, attention=True, attention_size=attention_size)

        init_ = lambda m: init(m, nn.init.xavier_uniform_, lambda x: nn.init.
                               constant_(x, 0))

        init_2 = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=1)

        self.exp_size = int(hidden_size / attention_size)

        self.input_attention = nn.Parameter(torch.ones(attention_size), requires_grad=True)
        self.ones = nn.Parameter(torch.ones(self.exp_size), requires_grad=False)

        self.main = nn.Sequential(
            ImpalaBlock(in_channels=num_inputs, out_channels=16),
            ImpalaBlock(in_channels=16, out_channels=32),
            ImpalaBlock(in_channels=32, out_channels=32), nn.ReLU(), Flatten(),
            init_(nn.Linear(in_features=32 * 8 * 8, out_features=256)), nn.ReLU())

        self.critic_linear = init_2(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks, attn_masks, reuse_masks=False):
        x = inputs
        probs = torch.sigmoid(self.input_attention)
        probs = torch.distributions.bernoulli.Bernoulli(probs=probs)
        new_attn = probs.sample([inputs.shape[0]])
        new_attn_masks = attn_masks if reuse_masks else new_attn
        attn_masks = new_attn_masks * (1 - masks) + attn_masks * masks
        attn_log_probs = torch.flatten(probs.log_prob(attn_masks), start_dim=1).sum(dim=1).reshape([inputs.shape[0], 1])
        m = torch.kron(attn_masks, self.ones)

        x = self.main(x)
        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        x = m * x

        return self.critic_linear(x), x, rnn_hxs, attn_log_probs, attn_masks

    def attn_log_probs(self, attn_masks):
        probs = torch.sigmoid(self.input_attention)
        probs = torch.distributions.bernoulli.Bernoulli(probs=probs)
        attn_log_probs = torch.flatten(probs.log_prob(attn_masks), start_dim=1).sum(dim=1).reshape(
            [attn_masks.shape[0], 1])

        return attn_log_probs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('tanh'))


        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks, att_mask, reuse_masks=False):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs, None, att_mask


class MLPAttnBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPAttnBase, self).__init__(recurrent, num_inputs, hidden_size)

        num_obs_input = num_inputs

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.input_attention = nn.Parameter(torch.zeros(num_obs_input), requires_grad=True)

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs
        x = F.softmax(self.input_attention, dim=0) * x

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs


class MLPHardAttnBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPHardAttnBase, self).__init__(recurrent, num_inputs, hidden_size)

        num_obs_input = num_inputs

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.input_attention = nn.Parameter(torch.zeros(num_obs_input), requires_grad=True)

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs
        probs = F.softmax(self.input_attention, dim=0)
        probs = probs / torch.max(probs)
        m_soft = RelaxedBernoulli(1.0, probs=probs).sample()
        m_hard = 0.5 * (torch.sign(m_soft - 0.5) + 1)
        mask = m_hard - m_soft.detach() + m_soft
        x = mask * x

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs


class MLPHardAttnReinforceBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPHardAttnReinforceBase, self).__init__(recurrent, num_inputs, hidden_size)

        num_obs_input = num_inputs

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.input_attention = nn.Parameter(torch.ones(num_obs_input), requires_grad=True)

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    # forward of the NN with attention. The trick is that we sometimes need the attention to be randomly drawn (first
    # step of the task), and sometimes we want to give it as input (remaining steps of task). We use masks for this,
    # which indicate the beginning of a new episode.
    def forward(self, inputs, rnn_hxs, masks, attn_masks, reuse_masks=False):
        x = inputs
        probs = torch.sigmoid(self.input_attention.repeat([inputs.shape[0], 1]))
        probs = torch.distributions.bernoulli.Bernoulli(probs=probs)
        new_attn_masks = attn_masks if reuse_masks else probs.sample()
        attn_masks = new_attn_masks * (1 - masks) + attn_masks * masks  # masks=1 in first step of episode
        attn_log_probs = probs.log_prob(attn_masks).sum(dim=1).reshape([inputs.shape[0], 1])
        x = attn_masks * x

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)
        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs, attn_log_probs, attn_masks

    def attn_log_probs(self, attn_masks):
        probs = torch.sigmoid(self.input_attention.repeat([attn_masks.shape[0], 1]))
        probs = torch.distributions.bernoulli.Bernoulli(probs=probs)
        attn_log_probs = probs.log_prob(attn_masks).sum(dim=1).reshape([attn_masks.shape[0], 1])
        return attn_log_probs
