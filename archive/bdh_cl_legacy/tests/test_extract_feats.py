import numpy as np
import pytest

torchvision = pytest.importorskip("torchvision")

from bdh_cl.scripts.extract_feats import backbone, encode


def test_encode_shape_and_determinism():
    m = backbone(weights=False)
    x = np.random.default_rng(0).integers(0, 256, (3, 3, 32, 32), dtype=np.uint8)
    a, b = encode(m, x, "cpu", bs=2, size=64), encode(m, x, "cpu", bs=3, size=64)
    assert a.shape == (3, 512)
    assert np.allclose(a.numpy(), b.numpy(), atol=1e-4)   # batching-invariant in eval mode
