
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

import math


LOG_2PI = math.log(2 * math.pi)

class Agent(nn.Module):
    def __init__(self, act_dim, obs_dim, args = {}):
        super().__init__()
        self.args = args

        self._critic = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),

            nn.LazyLinear(128),
            nn.Tanh(),

            nn.LazyLinear(1),
        )

        self._actor_mean = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),

            nn.LazyLinear(128),
            nn.Tanh(),

            nn.LazyLinear(act_dim),
        )

        self._actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, obs):
        return self._critic(obs)

    def get_action_and_value(self, obs, action=None):
        mean = self._actor_mean(obs)
        logstd = self._actor_logstd.expand_as(mean)
        std = torch.exp(logstd)

        if action is None:
            action = mean + std * torch.randn_like(mean)

        logprob = (-0.5 * ((action - mean) / std) ** 2 - logstd - 0.5 * LOG_2PI).sum(-1)
        entropy = (logstd + 0.5 * (1 + LOG_2PI)).sum(-1)
        value = self._critic(obs)
        return action, logprob, entropy, value


    def ppo_update(self, optimizer, batch):
        b_obs, b_actions, b_logprobs, b_advantages, b_returns, b_values = batch
        batch_size = b_obs.shape[0]
        minibatch_size = batch_size // self.args.minibatch
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        idxs = np.arange(batch_size)
        for _ in range(self.args.epochs):
            np.random.shuffle(idxs)
            for start in range(0, batch_size, minibatch_size):
                mb_idx = idxs[start:start + minibatch_size]

                _, newlogprob, entropy, newvalue = self.get_action_and_value(b_obs[mb_idx], b_actions[mb_idx])
                logratio = newlogprob - b_logprobs[mb_idx]
                ratio = logratio.exp()

                mb_adv = b_advantages[mb_idx]
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - self.args.clip, 1 + self.args.clip)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((newvalue.squeeze(-1) - b_returns[mb_idx]) ** 2).mean()
                entropy_loss = entropy.mean()

                loss = pg_loss - self.args.entr * entropy_loss + self.args.vf * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 0.1)
                optimizer.step()

        return pg_loss.item(), v_loss.item(), entropy_loss.item()
