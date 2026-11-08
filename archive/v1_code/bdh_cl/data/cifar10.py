"""Official CIFAR-10 archive, direct safe member reads and fixed holdout."""
import hashlib
import pickle
from pathlib import Path
import tarfile
import urllib.request
import numpy as np

URL='https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
MD5='c58f30108f718f92721af3b95e74349a'


def load(root='data_cache/cifar10',download=False):
    root=Path(root);archive=root/'cifar-10-python.tar.gz'
    if not archive.exists():
        if not download:raise FileNotFoundError(archive)
        root.mkdir(parents=True,exist_ok=True)
        temp=archive.with_suffix('.download')
        urllib.request.urlretrieve(URL,temp)
        temp.rename(archive)
    if hashlib.md5(archive.read_bytes()).hexdigest()!=MD5:raise ValueError('CIFAR-10 checksum mismatch')
    with tarfile.open(archive,'r:gz') as handle:
        def read(name):
            record=pickle.load(handle.extractfile('cifar-10-batches-py/'+name),encoding='bytes')
            return record[b'data'].reshape(-1,3,32,32),np.array(record[b'labels'],dtype=np.int64)
        batches=[read(f'data_batch_{i}') for i in range(1,6)]
        images=np.concatenate([r[0] for r in batches]);labels=np.concatenate([r[1] for r in batches])
        test_images,test_labels=read('test_batch')
    rng=np.random.default_rng(20261102)
    valid=np.concatenate([rng.permutation(np.flatnonzero(labels==c))[:500] for c in range(10)])
    fit=np.setdiff1d(np.arange(len(labels)),valid)
    return dict(images=images,labels=labels,test_images=test_images,test_labels=test_labels,train=fit,validation=valid)


if __name__=='__main__':
    data=load(download=True)
    print('CIFAR-10 verified:',len(data['train']),'fit,',len(data['validation']),'validation,',len(data['test_labels']),'test')
