
from math import inf
import argparse
import time
import torch
import numpy as np

import agents.ppo as ppo
import racing_env

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_AGENTS = 2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--envs", type=int, default=256)
    p.add_argument("--episodes", type=int, default=150)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--minibatch", type=int, default=64)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--entr", type=float, default=0.02)
    p.add_argument("--vf", type=float, default=0.2)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def collect_rollout(venv, agents, args):
    S, E = args.steps, args.envs
    obs_buf  = torch.zeros((S, E, venv.obs_dim()), device=device)
    act_buf  = torch.zeros((S, E, venv.act_dim()), device=device)
    logp_buf = torch.zeros((S, E), device=device)
    val_buf  = torch.zeros((S, E), device=device)
    rew_buf  = torch.zeros((S, E), device=device)
    term_buf = torch.zeros((S, E), device=device)
    trun_buf = torch.zeros((S, E), device=device)
    tau_buf  = torch.zeros((S, E), device=device)
    aid_buf  = torch.zeros((S, E), dtype=torch.long, device=device)

    next_obs = torch.tensor(np.asarray(venv.reset(), dtype=np.float32), device=device)
    next_aid = torch.tensor(np.asarray(venv.current_agents(), dtype=np.int64), device=device)

    for t in range(S):
        obs_buf[t] = next_obs
        aid_buf[t] = next_aid

        # query each agent only on the envs where it is the one acting now
        action = torch.zeros((E, venv.act_dim()), device=device)
        logp = torch.zeros(E, device=device)
        value = torch.zeros(E, device=device)
        for a in range(N_AGENTS):
            mask = (next_aid == a)
            if not mask.any():
                continue
            idx = mask.nonzero(as_tuple=True)[0]
            with torch.no_grad():
                act_a, lp_a, _, v_a = agents[a].get_action_and_value(next_obs[idx])
            action[idx] = act_a
            logp[idx]   = lp_a
            value[idx]  = v_a.squeeze(-1)

        act_buf[t]  = action
        logp_buf[t] = logp
        val_buf[t]  = value

        obs_np, rew, term, trun, tau = venv.step(action.cpu().double().numpy())
        next_obs = torch.tensor(obs_np, dtype=torch.float32, device=device)
        next_aid = torch.tensor(np.asarray(venv.current_agents(), dtype=np.int64), device=device)

        rew_buf[t] = torch.tensor(rew, device=device)
        term_buf[t] = torch.tensor(term, device=device)
        trun_buf[t] = torch.tensor(trun, device=device)
        tau_buf[t] = torch.tensor(tau, device=device)

        for name, x in [("obs", next_obs), ("rew", rew_buf[t]), ("tau", tau_buf[t])]:
            if not torch.isfinite(x).all():
                raise RuntimeError(f"non-finite {name} at step {t}")

    next_val = torch.zeros(E, device=device)
    for a in range(N_AGENTS):
        mask = (next_aid == a)
        if not mask.any():
            continue
        idx = mask.nonzero(as_tuple=True)[0]
        with torch.no_grad():
            next_val[idx] = agents[a].get_value(next_obs[idx]).squeeze(-1)

    adv_buf = torch.zeros((S, E), device=device)
    ret_buf = torch.zeros((S, E), device=device)

    aid_cpu = aid_buf.cpu().numpy()
    for a in range(N_AGENTS):
        last_val = next_val.clone()
        last_val[next_aid != a] = 0.0
        lastgae = torch.zeros(E, device=device)
        have_next = (next_aid == a)

        for t in reversed(range(S)):
            rows = torch.tensor((aid_cpu[t] == a), device=device)
            if not rows.any():
                continue

            nonterminal = 1.0 - term_buf[t]
            disc = args.gamma ** tau_buf[t]
            
            nextvalues = torch.where(have_next, last_val, torch.zeros_like(last_val))
            delta = rew_buf[t] + disc * nextvalues * nonterminal - val_buf[t]
            gae = delta + disc * args.gae_lambda * nonterminal * lastgae

            adv_buf[t] = torch.where(rows, gae, adv_buf[t])
            lastgae = torch.where(rows, gae, lastgae)
            last_val = torch.where(rows, val_buf[t], last_val)
            have_next = torch.where(rows, torch.ones_like(have_next, dtype=torch.bool), have_next)

    ret_buf = adv_buf + val_buf

    # split flat buffers by agent for separate updates
    out = {}
    flat_aid = aid_buf.reshape(-1)
    for a in range(N_AGENTS):
        m = (flat_aid == a)
        out[a] = (
            obs_buf.reshape(-1, venv.obs_dim())[m], act_buf.reshape(-1, venv.act_dim())[m],
            logp_buf.reshape(-1)[m], adv_buf.reshape(-1)[m], ret_buf.reshape(-1)[m], val_buf.reshape(-1)[m],
        )
    return out


def main(args):
    venv = racing_env.VecRacingEnv(args.seed, args.envs)

    agents = [ppo.Agent(venv.act_dim(), venv.obs_dim(), args).to(device) for _ in range(N_AGENTS)]
    opts = [torch.optim.Adam(a.parameters(), lr=args.lr, eps=1e-5) for a in agents]

    best_mean_return = -inf
    for update in range(1, args.episodes + 1):
        batch = collect_rollout(venv, agents, args)

        stats = []
        for a in range(N_AGENTS):
            pg, vl, ent = agents[a].ppo_update(opts[a], batch[a])
            stats.append((batch[a][4].mean().item(), pg, vl, ent))

        if update % 10 == 0:
            mean_ret_all = sum(s[0] for s in stats) / N_AGENTS
            mean_ent_all = sum(s[3] for s in stats) / N_AGENTS
            print(f"update {update} mean_return={mean_ret_all:.4f} entropy={mean_ent_all:.4f}")

        mean_return = sum(s[0] for s in stats) / N_AGENTS
        if best_mean_return < mean_return:
            best_mean_return = mean_return
            for a in range(N_AGENTS):
                torch.save(agents[a].state_dict(), f"models/racing_env/ppo_racer_agent_best_{a}.pt")

    print("Saved agents.")


if __name__ == "__main__":
    main(parse_args())