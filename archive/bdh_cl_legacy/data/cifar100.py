"""CIFAR-100 loading, deterministic disjoint class order and train/validation split."""
import hashlib
import pickle
import tarfile
import urllib.request
from pathlib import Path
import numpy as np

URL='https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
MD5='eb9058c3a382ffc7106e4002c42a8d85'


def prepare(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    archive=root/'cifar-100-python.tar.gz'
    if not archive.exists():
        tmp=archive.with_suffix('.download')
        urllib.request.urlretrieve(URL,tmp)
        tmp.rename(archive)
    if hashlib.md5(archive.read_bytes()).hexdigest()!=MD5:
        raise ValueError('CIFAR-100 archive checksum mismatch')
    return archive


def load(root,download=False):
    archive=prepare(root) if download else Path(root)/'cifar-100-python.tar.gz'
    # Read known members directly, avoiding filesystem extraction.
    with tarfile.open(archive,'r:gz') as tar:
        records=[]
        for name in ('train','test'):
            data=pickle.load(tar.extractfile('cifar-100-python/'+name),encoding='bytes')
            records.append((data[b'data'].reshape(-1,3,32,32),np.array(data[b'fine_labels'])))
    order=np.random.default_rng(2026).permutation(100)
    inverse=np.argsort(order)
    (images,labels),(test_images,test_labels)=records
    labels=inverse[labels];test_labels=inverse[test_labels]
    rng=np.random.default_rng(42);valid=[]
    for c in range(100):valid.extend(rng.permutation(np.flatnonzero(labels==c))[:50])
    valid=np.array(valid);train=np.setdiff1d(np.arange(len(labels)),valid)
    return (images[train],labels[train]),(images[valid],labels[valid]),(test_images,test_labels),order.tolist()

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args()
    print(prepare(a.root),flush=True)
