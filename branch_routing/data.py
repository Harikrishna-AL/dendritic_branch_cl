"""Split MNIST and Split CIFAR-10: five tasks of two classes each (task = label // 2).

Both loaders return a :class:`SplitData` with train / validation / test tensors.
MNIST images are standardised float vectors (N, 784). CIFAR-10 images stay uint8
(N, 3, 32, 32); models normalise them with :func:`normalize_cifar`.
"""
from __future__ import annotations

import gzip
import hashlib
import pickle
import struct
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

N_TASKS = 5
N_CLASSES = 10
REPO_ROOT = Path(__file__).resolve().parent.parent
MNIST_SPLIT = REPO_ROOT / 'data' / 'splits' / 'mnist_split_indices.npz'
MNIST_MEAN, MNIST_STD = .13062755, .30810780
CIFAR_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
CIFAR_MD5 = 'c58f30108f718f92721af3b95e74349a'
CIFAR_MEAN = torch.tensor([.4914, .4822, .4465])[None, :, None, None]
CIFAR_STD = torch.tensor([.247, .2435, .2616])[None, :, None, None]


def task_of(y: torch.Tensor) -> torch.Tensor:
    return y // 2


def task_order(first: int) -> list[int]:
    """Forward (first=0) or reverse (first=4) task order."""
    return list(range(N_TASKS)) if first == 0 else list(range(N_TASKS - 1, -1, -1))


@dataclass
class SplitData:
    name: str
    x: torch.Tensor
    y: torch.Tensor
    val_x: torch.Tensor
    val_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor

    def split(self, which: str) -> tuple[torch.Tensor, torch.Tensor]:
        return {'train': (self.x, self.y), 'val': (self.val_x, self.val_y), 'test': (self.test_x, self.test_y)}[which]


# ---------------------------------------------------------------- MNIST
def _read_idx(path: Path) -> np.ndarray:
    raw = gzip.open(path, 'rb').read()
    magic, count = struct.unpack('>II', raw[:8])
    if magic == 2051:
        rows, cols = struct.unpack('>II', raw[8:16])
        return np.frombuffer(raw[16:], dtype=np.uint8).reshape(count, rows, cols)
    if magic == 2049:
        return np.frombuffer(raw[8:], dtype=np.uint8)
    raise ValueError(f'{path.name}: unexpected IDX magic {magic}')


def load_mnist(root: str | Path = 'data_cache/mnist') -> SplitData:
    """55,000 train / 5,000 validation (fixed split file) / 10,000 test."""
    root = Path(root)
    files = {k: root / f'{v}-ubyte.gz' for k, v in dict(train_images='train-images-idx3', train_labels='train-labels-idx1',
                                                          test_images='t10k-images-idx3', test_labels='t10k-labels-idx1').items()}
    missing = [p.name for p in files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f'MNIST files missing from {root}: {missing}. Run: bash scripts/fetch_mnist.sh')
    raw = {k: _read_idx(p) for k, p in files.items()}
    split = np.load(MNIST_SPLIT)

    def prep(images, labels):
        x = (torch.tensor(images.copy()).float().reshape(-1, 784) / 255 - MNIST_MEAN) / MNIST_STD
        return x, torch.tensor(labels.copy(), dtype=torch.long)

    tr, va = split['train'], split['validation']
    return SplitData('mnist', *prep(raw['train_images'][tr], raw['train_labels'][tr]),
                     *prep(raw['train_images'][va], raw['train_labels'][va]), *prep(raw['test_images'], raw['test_labels']))


# ---------------------------------------------------------------- CIFAR-10
def normalize_cifar(x: torch.Tensor) -> torch.Tensor:
    return (x.float() / 255 - CIFAR_MEAN) / CIFAR_STD


def load_cifar10(root: str | Path = 'data_cache/cifar10', download: bool = False) -> SplitData:
    """45,000 train / 5,000 validation (500 per class, fixed seed) / 10,000 test."""
    root = Path(root); archive = root / 'cifar-10-python.tar.gz'
    if not archive.exists():
        if not download:
            raise FileNotFoundError(f'{archive} missing; call load_cifar10(download=True)')
        root.mkdir(parents=True, exist_ok=True); tmp = archive.with_suffix('.download')
        urllib.request.urlretrieve(CIFAR_URL, tmp); tmp.rename(archive)
    if hashlib.md5(archive.read_bytes()).hexdigest() != CIFAR_MD5:
        raise ValueError('CIFAR-10 checksum mismatch')
    with tarfile.open(archive, 'r:gz') as tar:
        def read(name):
            rec = pickle.load(tar.extractfile('cifar-10-batches-py/' + name), encoding='bytes')
            return rec[b'data'].reshape(-1, 3, 32, 32), np.array(rec[b'labels'], dtype=np.int64)
        batches = [read(f'data_batch_{i}') for i in range(1, 6)]
        images, labels = np.concatenate([b[0] for b in batches]), np.concatenate([b[1] for b in batches])
        test_images, test_labels = read('test_batch')
    rng = np.random.default_rng(20261102)
    val = np.concatenate([rng.permutation(np.flatnonzero(labels == c))[:500] for c in range(10)])
    train = np.setdiff1d(np.arange(len(labels)), val)
    t = lambda a: torch.from_numpy(np.array(a, copy=True))
    return SplitData('cifar10', t(images[train]), t(labels[train]).long(), t(images[val]), t(labels[val]).long(), t(test_images), t(test_labels).long())


def load(bench: str) -> SplitData:
    return {'mnist': load_mnist, 'cifar10': load_cifar10}[bench]()
