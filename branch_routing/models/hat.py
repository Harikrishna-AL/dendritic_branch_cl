"""HAT: Hard Attention to the Task (Serrà et al., ICML 2018), adapted as in the official code
(github.com/joansj/hat, networks/mlp_hat.py and alexnet_hat.py).

Each task has a learned embedding per gated layer; its unit mask is sigmoid(s * embedding).
Units used by earlier tasks are protected: gradients of weights between used units are
scaled by 1 - min(pre-mask, post-mask). Adaptation for class-incremental testing (as in the
earlier comparison in this project): ONE shared 10-class head instead of per-task heads.
The head is shared and not protected. A task must be given (or inferred) at test time.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ..data import N_TASKS, normalize_cifar


class HATNetwork(nn.Module):
    is_branch = False
    uses_task = True

    def masks(self, tasks: torch.Tensor, s: float) -> list[torch.Tensor]:
        """tasks: (N,) task indices -> per-layer (N, units) masks."""
        return [torch.sigmoid(s * e(tasks)) for e in self.embeddings]

    def embedding_parameters(self):
        return [e.weight for e in self.embeddings]


class HATMLP(HATNetwork):
    """784 -> width -> width (ReLU, masked) -> shared 10-class head. Native dropout .2 / .5."""

    def __init__(self, width: int, seed: int = 0, dropout: bool = True):
        super().__init__(); torch.manual_seed(seed); self.width = width
        self.fc1, self.fc2 = nn.Linear(784, width), nn.Linear(width, width); self.head = nn.Linear(width, 10)
        self.embeddings = nn.ModuleList([nn.Embedding(N_TASKS, width) for _ in range(2)])
        self.p_in, self.p_hid = (.2, .5) if dropout else (0., 0.)

    def forward(self, x, tasks, s: float = 400.):
        m1, m2 = self.masks(tasks, s)
        h = F.dropout(x, self.p_in, self.training)
        h = F.dropout(F.relu(self.fc1(h)), self.p_hid, self.training) * m1
        h = F.dropout(F.relu(self.fc2(h)), self.p_hid, self.training) * m2
        return self.head(h)

    def protection_views(self, used: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        """Per-parameter fraction of each weight that earlier tasks rely on (official get_view_for)."""
        u1, u2 = used
        return {'fc1.weight': u1[:, None].expand_as(self.fc1.weight), 'fc1.bias': u1,
                'fc2.weight': torch.min(u2[:, None], u1[None, :]).expand_as(self.fc2.weight), 'fc2.bias': u2}


def hat_mlp_width(n_params: int) -> int:
    """Width whose HATMLP (incl. task embeddings) has closest to n_params parameters."""
    return min(range(16, 600), key=lambda w: abs(w * w + 806 * w + 10 - n_params))


class HATCNN(HATNetwork):
    """Same layers as the ordinary CNN (tanh convs 32/64/128 -> dense 32 -> head) with a HAT
    mask on every conv channel and dense unit. No LayerNorm: it would mix masked and unmasked units."""

    def __init__(self, seed: int = 0, dropout: bool = True):
        super().__init__(); torch.manual_seed(seed)
        self.convs = nn.ModuleList([nn.Conv2d(a, b, 3, 2, 1) for a, b in zip([3, 32, 64], [32, 64, 128])])
        self.fc = nn.Linear(2048, 32); self.head = nn.Linear(32, 10)
        self.embeddings = nn.ModuleList([nn.Embedding(N_TASKS, n) for n in [32, 64, 128, 32]])
        self.p_in, self.p_hid = (.2, .5) if dropout else (0., 0.)

    def forward(self, x, tasks, s: float = 400.):
        ms = self.masks(tasks, s); h = F.dropout(normalize_cifar(x), self.p_in, self.training)
        for conv, m in zip(self.convs, ms[:3]): h = conv(h).tanh() * m[:, :, None, None]
        h = F.dropout(h.flatten(1), self.p_hid, self.training)
        return self.head(F.dropout(self.fc(h).tanh(), self.p_hid, self.training) * ms[3])

    def protection_views(self, used):
        u = used; views = {}
        for i, conv in enumerate(self.convs):
            post = u[i][:, None, None, None]
            views[f'convs.{i}.weight'] = (post if i == 0 else torch.min(post, u[i - 1][None, :, None, None])).expand_as(conv.weight)
            views[f'convs.{i}.bias'] = u[i]
        pre = u[2].repeat_interleave(16)  # conv3 channel mask over its 4x4 positions
        views['fc.weight'] = torch.min(u[3][:, None], pre[None, :]).expand_as(self.fc.weight); views['fc.bias'] = u[3]
        return views
