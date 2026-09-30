
import random

import numpy as np
import torch


def scalar(x) -> float:
    return float(np.asarray(x).reshape(-1)[0])


class EpisodeRunner:
    def __init__(self, spec, agents, device="cpu"):
        self.spec = spec
        self.agents = agents if isinstance(agents, (list, tuple)) else [agents]
        self.device = torch.device(device)
        self.reset()

    def reset(self):
        self.seed = random.randrange(1, 1_000_000)
        self.env = self.spec.make_single(self.seed)

        self.obs = np.asarray(self.env.reset(), dtype=np.float32).reshape(-1)
        self.raw = self.env.render_state()
        self.raw_prev = self.raw

        self.aid = self.spec.current_agent(self.env)
        self.tau = 0.5
        self.t = 0.0
        self.reward = 0.0
        self.last_rew = 0.0
        self.action = np.zeros(self.env.act_dim(), dtype=np.float64)
        self.n_dec = 0
        self.per_agent_dec = [0] * self.spec.n_agents
        self.done = False
        self.status = "RUN"

    def step(self) -> bool:
        self.aid = self.spec.current_agent(self.env)
        agent = self.agents[self.aid % len(self.agents)]

        obs_t = torch.as_tensor(self.obs, dtype=torch.float32,
                                device=self.device).unsqueeze(0)
        with torch.no_grad():
            act, *_ = agent.get_action_and_value(obs_t)

        act_np = act.cpu().double().numpy()
        self.action = act_np.reshape(-1)

        obs, rew, term, trun, tau = self.env.step(act_np)

        self.obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        self.last_rew = scalar(rew)
        self.reward += self.last_rew
        self.tau = max(scalar(tau), 1e-3)
        self.t += self.tau
        self.n_dec += 1
        self.per_agent_dec[self.aid] += 1

        terminated = bool(scalar(term))
        truncated = bool(scalar(trun))

        self.raw_prev = self.raw
        self.raw = self.env.render_state()

        if terminated or truncated:
            self.status = self.spec.status_fn(self.last_rew, terminated)
            self.done = True
            return False
        return True


class BaseRenderer:
    world_range = (-0.5, 5.5, -0.5, 5.5)
    lock_aspect = True

    def setup(self, plot, cfg):
        raise NotImplementedError

    def on_reset(self, run):
        pass

    def on_decision(self, run):
        pass

    def draw(self, run, frac):
        raise NotImplementedError

    def hud(self, run) -> str:
        return (f"t={run.t:5.2f}s  reward={run.reward:7.3f}  "
                f"tau={run.tau:4.2f}  [{run.status}]")