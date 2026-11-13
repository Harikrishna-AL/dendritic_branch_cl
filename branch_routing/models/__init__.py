from .baselines import CNN, MLP, mlp_width
from .branch_net import CNN_ARMS, BranchCNN, BranchMLP, BranchNetwork
from .layers import ConvBranchLayer, SparseBranchLayer, patch_indices


def build_model(bench: str, kind: str, seed: int, arch: str | None = None):
    """kind: 'branch' or 'plain'. arch: MNIST depth ('1', '2', ...) or a CNN arm name.
    Plain networks are matched in parameter count to the branch network."""
    if bench == 'mnist':
        depth = int(arch or 1)
        if kind == 'branch': return BranchMLP(depth, seed)
        n_params = sum(p.numel() for p in BranchMLP(depth, seed).parameters())
        return MLP(depth, mlp_width(depth, n_params), seed)
    if kind == 'branch': return BranchCNN(arch or 'conv_b4_sparse', seed)
    return CNN(seed)
from .hat import HATCNN, HATMLP, hat_mlp_width


def build_hat(bench: str, seed: int, dropout: bool = True):
    if bench == 'mnist':
        return HATMLP(hat_mlp_width(sum(p.numel() for p in BranchMLP(1, seed).parameters())), seed, dropout)
    return HATCNN(seed, dropout)
