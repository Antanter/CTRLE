
import argparse
import time

import torch
import numpy as np

import agents.ppo as ppo
import robot_env
from math import inf

torch.distributions.Distribution.set_default_validate_args(False)

import os
print("PID:", os.getpid())


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--envs", type=int, default=1024)
    p.add_argument("--episodes", type=int, default=700)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--minibatch", type=int, default=64)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--entr", type=float, default=0.005)
    p.add_argument("--vf", type=float, default=0.2)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


start = time.time()


def collect_rollout(venv, agent, args):
    obs_buf = torch.zeros((args.steps, args.envs, venv.obs_dim()), device=device)
    act_buf = torch.zeros((args.steps, args.envs, venv.act_dim()), device=device)
    logprob_buf = torch.zeros((args.steps, args.envs), device=device)
    rew_buf = torch.zeros((args.steps, args.envs), device=device)
    term_buf = torch.zeros((args.steps, args.envs), device=device)
    trun_buf = torch.zeros((args.steps, args.envs), device=device)
    val_buf = torch.zeros((args.steps, args.envs), device=device)
    tau_buf = torch.zeros((args.steps, args.envs), device=device)

    next_obs = torch.tensor(np.asarray(venv.reset(), dtype=np.float32), device=device)

    for t in range(args.steps):
        obs_buf[t] = next_obs
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_obs)
        act_buf[t] = action
        logprob_buf[t] = logprob
        val_buf[t] = value.squeeze(-1)

        obs_np, rewards, term, trun, taus = venv.step(action.cpu().double().numpy())

        next_obs = torch.tensor(obs_np, dtype=torch.float32, device=device)
        rew_buf[t] = torch.tensor(rewards, device=device)
        term_buf[t] = torch.tensor(term, device=device)
        trun_buf[t] = torch.tensor(trun, device=device)
        tau_buf[t] = torch.tensor(taus, device=device)

    with torch.no_grad():
        next_value = agent.get_value(next_obs).squeeze(-1)

    term_mask = term_buf.bool()
    trun_mask = trun_buf.bool()
    success_mask = term_mask & (rew_buf > 0.5)
    crash_mask   = term_mask & (rew_buf <= 0.5)

    n_success = success_mask.sum().item()
    n_crash   = crash_mask.sum().item()
    n_timeout = trun_mask.sum().item()
    n_episodes = n_success + n_crash + n_timeout

    stats = {
        "success_rate": n_success / max(n_episodes, 1),
        "crash_rate":   n_crash / max(n_episodes, 1),
        "timeout_rate": n_timeout / max(n_episodes, 1),
        "n_episodes":   n_episodes,
    }

    advantages = torch.zeros_like(rew_buf, device=device)
    lastgaelam = torch.zeros(args.envs, device=device)
    for t in reversed(range(args.steps)):
        nextnonterminal = 1.0 - (term_buf[t])
        nextvalues = next_value if t == args.steps - 1 else val_buf[t + 1]
        disc = args.gamma ** tau_buf[t]
        delta = rew_buf[t] + disc * nextvalues * nextnonterminal - val_buf[t]
        advantages[t] = lastgaelam = delta + disc * args.gae_lambda * nextnonterminal * lastgaelam

    returns = advantages + val_buf

    return (
        obs_buf.reshape(-1, venv.obs_dim()),
        act_buf.reshape(-1, venv.act_dim()),
        logprob_buf.reshape(-1),
        advantages.reshape(-1),
        returns.reshape(-1),
        val_buf.reshape(-1),
        stats,
    )

def main(args):
    venv = robot_env.VecRobotEnv(args.seed, args.envs)

    agent = ppo.Agent(venv.act_dim(), venv.obs_dim(), args).to(device)
    compiled = torch.compile(agent)

    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)

    best_mean_return = -inf
    for update in range(1, args.episodes + 1):
        batch = collect_rollout(venv, compiled, args)
        stats = batch[-1]
        pg_loss, v_loss, ent = compiled.ppo_update(optimizer, batch[:-1])
        mean_return = batch[4].mean().item()

        if update % 10 == 0:
            print(f"update {update:4d} | mean_return={mean_return:8.3f} | pg_loss={pg_loss:.4f} | v_loss={v_loss:.4f} "
                f"| entropy={ent:.4f} | success={stats['success_rate']:.3f} | crash={stats['crash_rate']:.3f} "
                f"| timeout={stats['timeout_rate']:.3f} | n_ep={stats['n_episodes']} | elapsed={time.time() - start:.1f}s")
            torch.save(agent.state_dict(), f"models/robot_env/ppo_smdp_agent_{update}.pt")
        
        if best_mean_return < mean_return:
            best_mean_return = mean_return
            torch.save(agent.state_dict(), f"models/robot_env/ppo_smdp_agent_best.pt")


if __name__ == "__main__":
    args = parse_args()

    main(args)