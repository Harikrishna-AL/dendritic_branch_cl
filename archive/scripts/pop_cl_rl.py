"""Exp22: continual RL -- one DQN agent cycles through the MinAtar games.

Why: on pretrained class-incremental vision the analytic readouts close the gap
(Exp21), so there is no room for a mechanism. Continual RL with no pretrained
features is where networks are known to *lose plasticity* (Dohare et al. 2024;
Sokar et al. 2023 ReDo; Abbas et al. 2023). The population layer keeps a usage
count per neuron; that count can drive both protection (tiredness: heavily used
neurons get small gradients) and recruitment (silent neurons are re-drawn).

Protocol: games are played in a fixed order, ``--cycles`` times; each game is a
phase of ``--steps`` env steps. The replay buffer is emptied and epsilon re-
annealed at every switch; the network, optimiser state and target net carry
over. The observation is the MinAtar state padded to 10 channels, so channel
meaning changes between games (the input distribution is non-stationary).

Measured per phase:
  train_return    mean return of episodes finished in the last 25% of the phase
                  -> plasticity: cycle-2 vs cycle-1 on the same game (and vs a
                  fresh agent: run ``--games <g> --cycles 1``)
  eval            greedy (eps 0.05) return on EVERY game at phase end -> forgetting
  dormant         fraction of hidden units that are silent (ReLU: ReDo score <= 0.1;
                  pop: fired <= 0.1 x expected rate over the last window)

Arms: dqn, dqn_redo (ReDo reset), pop, pop_tired, pop_reset, pop_tired_reset.

    python bdh_cl/scripts/pop_cl_rl.py --arm dqn --seed 0 --out results/exp22/dqn_s0.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bdh_cl.scripts.pop_cl_cifar import PopLayer, scale_grad  # noqa: E402

GAMES = ["breakout", "space_invaders", "freeway", "asterix", "seaquest"]
ARMS = ["dqn", "dqn_redo", "pop", "pop_tired", "pop_reset", "pop_tired_reset", "pop_homeo", "pop_homeo_up",
        "pop_graded", "pop_gradedst",
        "pop_graded_tired", "pop_graded_reset", "pop_graded_tired_reset"]
CH, N_ACTIONS = 10, 6


def make_env(game, seed):
    from minatar import Environment

    env = Environment(game)
    env.seed(seed)
    return env


def obs(env) -> np.ndarray:
    """MinAtar state (10, 10, C) bool -> (10, 10, 10) uint8, channels zero-padded."""
    s = env.state()
    o = np.zeros((10, 10, CH), np.uint8)
    o[..., : s.shape[2]] = s
    return o


def to_t(o, device):
    return torch.as_tensor(o, device=device).permute(0, 3, 1, 2).float()


# ------------------------------------------------------------------ model

class HomeoPop(PopLayer):
    """PopLayer + homeostatic excitability (intrinsic plasticity).

    Each neuron has an offset ``e`` (in z-score units) added to its score before
    the top-k, and a running firing rate ``r``. Every training forward:
        r <- (1 - alpha) r + alpha * (fraction of the batch it fired on)
        e <- e + eta * (k/N - r)
    so a neuron that rarely wins becomes easier to fire until it takes part.
    ``up_only``: e is clamped at >= 0 -- idle neurons are boosted, busy ones are
    never suppressed (so old winners keep firing on old inputs).
    """

    def __init__(self, d, n, branches, k, temp, eta=0.05, alpha=1e-3, up_only=False):
        super().__init__(d, n, branches, k, temp)
        self.eta, self.alpha, self.up_only = eta, alpha, up_only
        self.register_buffer("e", torch.zeros(n))
        self.register_buffer("r", torch.full((n,), k / n))

    def forward(self, h):
        score = torch.einsum("bd,nkd->bnk", h, self.w).add(self.b).amax(-1)
        mu, sd = score.mean(1, keepdim=True).detach(), score.std(1, keepdim=True).detach()
        z = (score - mu) / (sd + 1e-5) + self.e
        top = z.detach().topk(self.k + 1, dim=1).values
        hard = (z >= top[:, self.k - 1 : self.k]).float()
        theta = 0.5 * (top[:, self.k - 1] + top[:, self.k])
        soft = torch.sigmoid((z - theta[:, None]) / self.temp)
        a = hard + soft - soft.detach()
        if self.training:
            with torch.no_grad():
                self.fired_task += hard.sum(0)
                self.r.mul_(1 - self.alpha).add_(self.alpha * hard.mean(0))
                self.e.add_(self.eta * (self.k / self.e.numel() - self.r))
                if self.up_only:
                    self.e.clamp_(min=0)
            if self.tired:
                a = scale_grad(a, self.grad_scale)
        return a


class GradedPop(PopLayer):
    """Same top-k winners as PopLayer, but a winner outputs HOW FAR it is above
    the cut-off (z - theta > 0) instead of 1. Losers output 0.

    ``surrogate=False``: gradient flows through the winners' own scores (a k-winners
    ReLU) -- changes both the forward magnitude and the gradient.
    ``surrogate=True``: graded forward, but the backward pass is PopLayer's sigmoid
    surrogate -- isolates the effect of the forward magnitude alone.
    """

    def __init__(self, d, n, branches, k, temp, surrogate=False):
        super().__init__(d, n, branches, k, temp)
        self.surrogate = surrogate

    def forward(self, h):
        score = torch.einsum("bd,nkd->bnk", h, self.w).add(self.b).amax(-1)
        mu, sd = score.mean(1, keepdim=True).detach(), score.std(1, keepdim=True).detach()
        z = (score - mu) / (sd + 1e-5)
        top = z.detach().topk(self.k + 1, dim=1).values
        hard = (z >= top[:, self.k - 1 : self.k]).float()
        theta = 0.5 * (top[:, self.k - 1] + top[:, self.k])
        graded = hard * (z - theta[:, None])
        if self.surrogate:
            soft = torch.sigmoid((z - theta[:, None]) / self.temp)
            a = graded.detach() + soft - soft.detach()
        else:
            a = graded
        if self.training:
            self.fired_task += hard.detach().sum(0)
            if self.tired:
                a = scale_grad(a, self.grad_scale)
        return a


class QNet(nn.Module):
    def __init__(self, arm, width=512, branches=4, k=32, temp=0.5, homeo_eta=0.05, homeo_alpha=1e-3):
        super().__init__()
        self.conv = nn.Conv2d(CH, 16, 3)
        d = 16 * 8 * 8
        self.is_pop = arm.startswith("pop")
        if self.is_pop:
            if "homeo" in arm:
                self.hidden = HomeoPop(d, width, branches, k, temp, homeo_eta, homeo_alpha, up_only=arm.endswith("_up"))
            elif "graded" in arm:
                self.hidden = GradedPop(d, width, branches, k, temp, surrogate=arm.endswith("st"))
            else:
                self.hidden = PopLayer(d, width, branches, k, temp)
            self.hidden.tired = "tired" in arm
        else:
            self.hidden = nn.Sequential(nn.Linear(d, width), nn.ReLU())
        self.head = nn.Linear(width, N_ACTIONS)

    def features(self, x):
        return F.relu(self.conv(x)).flatten(1)

    def forward(self, x):
        return self.head(self.hidden(self.features(x)))


def _zero_adam_rows(opt, p, rows=None, cols=None):
    st = opt.state.get(p)
    if not st:
        return
    for key in ("exp_avg", "exp_avg_sq"):
        if rows is not None:
            st[key][rows] = 0
        if cols is not None:
            st[key][:, cols] = 0


@torch.no_grad()
def dormant_mlp(net, x, thresh):
    """ReDo score: mean |activation| normalised by the layer mean."""
    h = net.hidden(net.features(x))
    s = h.abs().mean(0)
    return s / (s.mean() + 1e-8) <= thresh


@torch.no_grad()
def reset_mlp(net, opt, dead):
    """ReDo: re-draw incoming weights of dead units, zero their outgoing weights."""
    lin = net.hidden[0]
    fresh = nn.Linear(lin.in_features, lin.out_features).to(lin.weight.device)
    lin.weight[dead] = fresh.weight[dead]
    lin.bias[dead] = fresh.bias[dead]
    net.head.weight[:, dead] = 0
    _zero_adam_rows(opt, lin.weight, rows=dead)
    _zero_adam_rows(opt, lin.bias, rows=dead)
    _zero_adam_rows(opt, net.head.weight, cols=dead)


@torch.no_grad()
def reset_pop(net, opt, dead):
    """Re-draw a silent neuron's branches, zero its outgoing weights, and give it
    back full plasticity (its usage history is cleared)."""
    pop = net.hidden
    d = pop.w.shape[-1]
    pop.w[dead] = torch.randn_like(pop.w[dead]) / d ** 0.5
    pop.b[dead] = 0
    net.head.weight[:, dead] = 0
    pop.fired_prev[dead] = 0
    pop.grad_scale[dead] = 1
    _zero_adam_rows(opt, pop.w, rows=dead)
    _zero_adam_rows(opt, pop.b, rows=dead)
    _zero_adam_rows(opt, net.head.weight, cols=dead)


# ----------------------------------------------------------------- replay

class Replay:
    def __init__(self, cap):
        self.s = np.zeros((cap, 10, 10, CH), np.uint8)
        self.s2 = np.zeros_like(self.s)
        self.a = np.zeros(cap, np.int64)
        self.r = np.zeros(cap, np.float32)
        self.d = np.zeros(cap, np.float32)
        self.cap, self.i, self.n = cap, 0, 0

    def add(self, s, a, r, s2, d):
        self.s[self.i], self.a[self.i], self.r[self.i], self.s2[self.i], self.d[self.i] = s, a, r, s2, d
        self.i = (self.i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, b, rng, device):
        j = rng.integers(0, self.n, b)
        t = lambda v: torch.as_tensor(v[j], device=device)  # noqa: E731
        return to_t(self.s[j], device), t(self.a), t(self.r), to_t(self.s2[j], device), t(self.d)


# ------------------------------------------------------------------- eval

@torch.no_grad()
def evaluate(net, game, episodes, seed, device, eps=0.05, max_steps=10000):
    """Greedy return, episodes run side by side so each step is one forward."""
    net.eval()
    rng = np.random.default_rng(seed)
    envs = [make_env(game, seed + i) for i in range(episodes)]
    for e in envs:
        e.reset()
    ret, alive = np.zeros(episodes), np.ones(episodes, bool)
    for _ in range(max_steps):
        idx = np.flatnonzero(alive)
        if not len(idx):
            break
        q = net(to_t(np.stack([obs(envs[i]) for i in idx]), device))
        greedy = q.argmax(1).cpu().numpy()
        for i, g in zip(idx, greedy):
            a = rng.integers(N_ACTIONS) if rng.random() < eps else g
            r, done = envs[i].act(a)
            ret[i] += r
            alive[i] = not done
    net.train()
    return float(ret.mean())


# ------------------------------------------------------------------- main

def run(a):
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    games = a.games.split(",")
    net = QNet(a.arm, a.width, a.branches, a.k, a.temp, a.homeo_eta, a.homeo_alpha).to(device)
    target = copy.deepcopy(net).eval()
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, eps=a.adam_eps)
    # expected firings of one neuron over a phase, if usage were uniform
    expected_phase = a.k / a.width * a.batch * max(0, (a.steps - a.learn_start)) / a.train_freq
    tau = a.tau_games * expected_phase
    wb = None
    if not a.no_wandb:
        try:
            import wandb

            wb = wandb.init(project="bdh-d", group=a.group, name=f"{a.tag or a.arm}_s{a.seed}", config=vars(a))
        except Exception as e:  # noqa: BLE001
            print("wandb off:", e)
    log, updates, t0 = [], 0, time.time()
    for phase in range(len(games) * a.cycles):
        game, cycle = games[phase % len(games)], phase // len(games)
        env = make_env(game, a.seed * 1000 + phase)
        env.reset()
        buf = Replay(min(a.buffer, a.steps))
        s, ep_ret, finished = obs(env), 0.0, []
        dormant, resets, fired_snap = [], 0, None
        if net.is_pop:
            fired_snap = net.hidden.fired_task.clone()
        for t in range(a.steps):
            eps = max(a.eps_end, 1 - (1 - a.eps_end) * t / a.eps_steps)
            if t < a.learn_start or rng.random() < eps:
                act = int(rng.integers(N_ACTIONS))
            else:
                net.eval()
                with torch.no_grad():
                    act = int(net(to_t(s[None], device)).argmax(1))
                net.train()
            r, done = env.act(act)
            s2 = obs(env)
            buf.add(s, act, r, s2, float(done))
            ep_ret += r
            if done:
                finished.append((t, ep_ret))
                env.reset()
                s, ep_ret = obs(env), 0.0
            else:
                s = s2
            if t >= a.learn_start and t % a.train_freq == 0:
                bs, ba, br, bs2, bd = buf.sample(a.batch, rng, device)
                q = net(bs).gather(1, ba[:, None]).squeeze(1)
                with torch.no_grad():
                    tgt = br + a.gamma * (1 - bd) * target(bs2).amax(1)
                loss = F.smooth_l1_loss(q, tgt)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                updates += 1
                if updates % a.target_every == 0:
                    target.load_state_dict(net.state_dict())
                if updates % a.redo_every == 0:
                    if net.is_pop:
                        win = net.hidden.fired_task - fired_snap
                        dead = win <= a.redo_thresh * a.k / a.width * a.batch * a.redo_every
                        fired_snap = net.hidden.fired_task.clone()
                    else:
                        dead = dormant_mlp(net, buf.sample(256, rng, device)[0], a.redo_thresh)
                    dormant.append(float(dead.float().mean()))
                    if (a.arm == "dqn_redo" or "reset" in a.arm) and dead.any():
                        (reset_pop if net.is_pop else reset_mlp)(net, opt, dead)
                        resets += int(dead.sum())
        tail = [rt for (tt, rt) in finished if tt >= 0.75 * a.steps]
        rec = {"phase": phase, "game": game, "cycle": cycle,
               "train_return": float(np.mean(tail)) if tail else float("nan"),
               "episodes": len(finished),
               "curve": [float(np.mean([rt for (tt, rt) in finished if lo <= tt < lo + a.curve_bin] or [np.nan]))
                         for lo in range(0, a.steps, a.curve_bin)],
               "dormant_mean": float(np.mean(dormant)) if dormant else float("nan"),
               "dormant_last": dormant[-1] if dormant else float("nan"),
               "resets": resets}
        if net.is_pop:
            fired_phase = net.hidden.fired_task.clone()
            tired = net.hidden.end_task(tau)
            rec["tired_mean"] = float(tired.mean())
            if isinstance(net.hidden, HomeoPop):
                rec["excit_mean"] = float(net.hidden.e.mean())
                rec["excit_max"] = float(net.hidden.e.max())
            rec["tired_frac_gt0.9"] = float((tired > 0.9).float().mean())
            rec["silent_phase_frac"] = float((fired_phase == 0).float().mean())
        rec["eval"] = {g: evaluate(net, g, a.eval_episodes, 10_000 + a.seed * 100 + phase, device)
                       for g in GAMES if g in games} if a.eval_episodes else {}
        rec["elapsed_s"] = round(time.time() - t0)
        log.append(rec)
        print(f"PHASE {phase} {game} c{cycle} train={rec['train_return']:.2f} dormant={rec['dormant_last']:.3f} "
              f"resets={resets} eval={ {g: round(v, 2) for g, v in rec['eval'].items()} } {rec['elapsed_s']}s", flush=True)
        if wb:
            wb.log({"phase": phase, f"train/{game}": rec["train_return"], "dormant": rec["dormant_last"],
                    **{f"eval/{g}": v for g, v in rec["eval"].items()}})
    summary = {}
    for g in games:
        per_cycle = [r["train_return"] for r in log if r["game"] == g]
        summary[g] = {"train_by_cycle": per_cycle,
                      "final_eval": log[-1]["eval"].get(g, float("nan"))}
    out = {"args": vars(a), "log": log, "summary": summary}
    print("SUMMARY", json.dumps(summary), flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    if wb:
        wb.finish()
    return out


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--games", default=",".join(GAMES))
    p.add_argument("--cycles", type=int, default=2)
    p.add_argument("--steps", type=int, default=200_000, help="env steps per phase")
    p.add_argument("--buffer", type=int, default=100_000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--adam-eps", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--learn-start", type=int, default=5000)
    p.add_argument("--train-freq", type=int, default=1)
    p.add_argument("--target-every", type=int, default=1000)
    p.add_argument("--eps-steps", type=int, default=50_000)
    p.add_argument("--eps-end", type=float, default=0.1)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--branches", type=int, default=4)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--temp", type=float, default=0.5)
    p.add_argument("--homeo-eta", type=float, default=0.05, help="excitability step per update")
    p.add_argument("--homeo-alpha", type=float, default=1e-3, help="firing-rate EMA rate per update")
    p.add_argument("--tau-games", type=float, default=1.0)
    p.add_argument("--redo-every", type=int, default=1000, help="updates between dormancy checks")
    p.add_argument("--redo-thresh", type=float, default=0.1)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--curve-bin", type=int, default=10_000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--group", default="exp22_continual_rl")
    p.add_argument("--tag", default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--out", default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse())
