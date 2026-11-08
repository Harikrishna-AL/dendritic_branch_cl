"""The slow synaptic state, and the properties that would silently void it.

The dangerous failures here are not crashes. A memory that leaks the current
batch into its own read would inflate every number; one that is never written,
or that vanishes on reload, would produce a clean null that looks like evidence
against the idea. Each of those gets a test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bdh_cl.config import BDHSpec  # noqa: E402
from bdh_cl.models.bdh_wrapper import BDHWrapper  # noqa: E402
from bdh_cl.models.persistent_memory import MemorySpec, PersistentMemory  # noqa: E402


def model(mem: dict | None = None, **kw):
    return BDHWrapper(BDHSpec(
        n_layer=kw.pop("n_layer", 2), n_embd=32, n_head=2,
        mlp_internal_dim_multiplier=2, dropout=0.0,
        persistent_memory=mem, **kw,
    ))


# -- off by default -------------------------------------------------------


def test_absent_by_default_and_bit_identical():
    """Every earlier result and the upstream-equivalence proof must be safe."""
    torch.manual_seed(0)
    a = model(None).eval()
    torch.manual_seed(0)
    b = model({"enabled": False}).eval()
    assert a.memory is None and b.memory is None
    x = torch.randint(0, 256, (2, 12))
    assert torch.equal(a(x, None)[0], b(x, None)[0])


# -- the state itself -----------------------------------------------------


def write(m, code, value):
    """stage + commit, i.e. one forward's worth of writing."""
    m.stage(code, value)
    m.commit()


def test_write_then_read_recalls_something():
    m = PersistentMemory(MemorySpec(enabled=True, decay=1.0, eta=1.0), 2, 4, 8)
    code = torch.zeros(1, 2, 3, 4)
    code[..., 0] = 1.0
    value = torch.randn(1, 2, 3, 8)
    # empty memory recalls nothing
    assert float(m.read(code, m.snapshot()).abs().max()) == 0.0
    write(m, code, value)
    assert float(m.read(code, m.snapshot()).abs().max()) > 0.0


def test_snapshot_is_not_a_view():
    """If the snapshot shared storage with sigma, the in-place commit would bump
    its version and backward would fail -- the original bug."""
    m = PersistentMemory(MemorySpec(enabled=True, decay=1.0, eta=1.0), 1, 3, 4)
    snap = m.snapshot()
    write(m, torch.ones(1, 1, 1, 3), torch.ones(1, 1, 1, 4))
    assert float(snap.abs().max()) == 0.0, "snapshot tracked sigma's mutation"
    assert snap.data_ptr() != m.sigma.data_ptr()


def test_decay_forgets_geometrically():
    m = PersistentMemory(MemorySpec(enabled=True, decay=0.5, eta=1.0), 1, 3, 4)
    code = torch.ones(1, 1, 1, 3)
    write(m, code, torch.ones(1, 1, 1, 4))
    first = m.sigma.clone()
    for _ in range(3):
        write(m, torch.zeros_like(code), torch.zeros(1, 1, 1, 4))
    assert torch.allclose(m.sigma, first * 0.125, atol=1e-6)


def test_decay_one_never_forgets():
    m = PersistentMemory(MemorySpec(enabled=True, decay=1.0, eta=1.0), 1, 3, 4)
    code = torch.ones(1, 1, 1, 3)
    write(m, code, torch.ones(1, 1, 1, 4))
    before = m.sigma.clone()
    for _ in range(5):
        write(m, torch.zeros_like(code), torch.zeros(1, 1, 1, 4))
    assert torch.allclose(m.sigma, before)


def test_write_magnitude_is_independent_of_batch_and_length():
    """eta must mean the same thing across configurations, or a sweep over it
    is really a sweep over batch size."""
    a = PersistentMemory(MemorySpec(enabled=True, decay=1.0, eta=1.0), 1, 3, 4)
    b = PersistentMemory(MemorySpec(enabled=True, decay=1.0, eta=1.0), 1, 3, 4)
    write(a, torch.ones(1, 1, 2, 3), torch.ones(1, 1, 2, 4))
    write(b, torch.ones(8, 1, 16, 3), torch.ones(8, 1, 16, 4))
    assert torch.allclose(a.sigma, b.sigma, atol=1e-5)


def test_write_every_slows_the_state():
    m = PersistentMemory(MemorySpec(enabled=True, decay=1.0, eta=1.0, write_every=3), 1, 2, 2)
    for _ in range(2):
        write(m, torch.ones(1, 1, 1, 2), torch.ones(1, 1, 1, 2))
    assert float(m.sigma.abs().max()) == 0.0    # first two writes skipped
    write(m, torch.ones(1, 1, 1, 2), torch.ones(1, 1, 1, 2))
    assert float(m.sigma.abs().max()) > 0.0


def test_reset_clears():
    m = PersistentMemory(MemorySpec(enabled=True), 1, 3, 4)
    write(m, torch.ones(1, 1, 1, 3), torch.ones(1, 1, 1, 4))
    m.reset()
    assert float(m.sigma.abs().max()) == 0.0


# -- the properties that would void the experiment ------------------------


@pytest.mark.parametrize("n_layer", [1, 2, 6])
def test_train_forward_then_backward_does_not_raise(n_layer):
    """Regression: every memory-enabled run died on the first backward pass.

    ``read`` consumed ``sigma`` inside the autograd graph and ``write`` then
    mutated that same storage in place during the same forward. ``no_grad``
    stops the write being TRACKED but does not stop the version counter
    advancing, so backward raised "one of the variables needed for gradient
    computation has been modified by an inplace operation".

    Parametrised over depth because the original bug needed two reads of the
    same state to show up, and a single-layer test would have passed.
    """
    m = model({"enabled": True, "decay": 0.99, "eta": 1.0}, n_layer=n_layer).train()
    x = torch.randint(0, 256, (2, 12))
    for _ in range(3):                     # several steps: state is non-empty
        _logits, loss = m(x, x)
        loss.backward()
        m.zero_grad(set_to_none=True)
    assert float(m.memory.sigma.abs().max()) > 0


def test_every_layer_reads_the_same_snapshot():
    """Writes must not become visible to later layers within one forward.

    BDH reuses one parameter set for all layers, so a live-mutating state would
    let layer 2 read layer 1's write from the SAME batch. Commit happens once,
    after the loop, so a forward over n layers advances the write counter once.
    """
    m = model({"enabled": True, "decay": 1.0, "eta": 1.0}, n_layer=6).train()
    x = torch.randint(0, 256, (2, 12))
    m(x, None)
    assert int(m.memory.writes) == 1, f"committed {int(m.memory.writes)} times, want 1"


def test_read_is_causal_within_a_forward():
    """The read must use the state from BEFORE this batch.

    If the write landed first, the model would be reading a memory that already
    contains the current batch -- leaking the target into the prediction and
    inflating every number produced.
    """
    m = model({"enabled": True, "decay": 1.0, "eta": 1.0}).train()
    x = torch.randint(0, 256, (2, 12))
    assert float(m.memory.sigma.abs().max()) == 0.0
    # With an empty state the read contributes nothing, so a training forward
    # must match an eval forward that also has an empty state.
    torch.manual_seed(0)
    out_train = m(x, None)[0]
    assert float(m.memory.sigma.abs().max()) > 0.0, "nothing was written"
    m2 = model({"enabled": True, "decay": 1.0, "eta": 1.0}).eval()
    m2.load_state_dict({k: v for k, v in m.state_dict().items() if "sigma" not in k and "writes" not in k}, strict=False)
    m2.memory.reset()
    torch.manual_seed(0)
    assert torch.allclose(out_train, m2(x, None)[0], atol=1e-5)


def test_no_write_in_eval_mode():
    """Evaluation must not modify what the model remembers."""
    m = model({"enabled": True}).eval()
    before = m.memory.sigma.clone()
    m(torch.randint(0, 256, (2, 12)), None)
    assert torch.equal(m.memory.sigma, before)


def test_memory_receives_no_gradient_but_encoders_still_do():
    """Sigma is Hebbian: written under no_grad, never a parameter. Gradients
    must still reach the encoders THROUGH the read, or the model cannot learn
    to use what it remembers."""
    m = model({"enabled": True, "decay": 1.0, "eta": 1.0}).train()
    x = torch.randint(0, 256, (2, 12))
    m(x, None)                       # populate the state
    assert not m.memory.sigma.requires_grad
    names = {n for n, _ in m.named_parameters()}
    assert not any("memory" in n for n in names), "Sigma must not be a parameter"
    _logits, loss = m(x, x)
    loss.backward()
    assert m.core.encoder.grad is not None
    assert float(m.core.encoder.grad.abs().sum()) > 0


def test_memory_survives_a_state_dict_round_trip():
    """A checkpoint of a model with memory has to include what it remembers, or
    reloading silently discards the thing under test."""
    m = model({"enabled": True, "decay": 1.0, "eta": 1.0}).train()
    m(torch.randint(0, 256, (2, 12)), None)
    assert float(m.memory.sigma.abs().max()) > 0
    m2 = model({"enabled": True, "decay": 1.0, "eta": 1.0})
    m2.load_state_dict(m.state_dict())
    assert torch.allclose(m2.memory.sigma, m.memory.sigma)


def test_memory_changes_the_output_once_populated():
    """Guards against a silent no-op: a memory that is written but never
    actually reaches the computation would produce a clean null result."""
    m = model({"enabled": True, "decay": 1.0, "eta": 5.0}).train()
    x = torch.randint(0, 256, (2, 12))
    m.eval()
    empty = m(x, None)[0].clone()
    m.train()
    for _ in range(3):
        m(x, None)
    m.eval()
    populated = m(x, None)[0]
    assert not torch.allclose(empty, populated, atol=1e-4)


def test_memory_is_position_free():
    """BDH RoPEs Q and K by absolute position. A state carried across sequences
    must not, or it mixes incompatible rotation frames. The same content at a
    different offset must therefore write the same thing.
    """
    m = PersistentMemory(MemorySpec(enabled=True, decay=1.0, eta=1.0), 1, 4, 4)
    code = torch.zeros(1, 1, 4, 4)
    code[0, 0, 0, 0] = 1.0                 # content at position 0
    val = torch.zeros(1, 1, 4, 4)
    val[0, 0, 0, 1] = 1.0
    write(m, code, val)
    a = m.sigma.clone()

    m.reset()
    shifted_code = torch.zeros(1, 1, 4, 4)
    shifted_code[0, 0, 3, 0] = 1.0         # same content at position 3
    shifted_val = torch.zeros(1, 1, 4, 4)
    shifted_val[0, 0, 3, 1] = 1.0
    write(m, shifted_code, shifted_val)
    assert torch.allclose(a, m.sigma, atol=1e-6)


# -- spec validation ------------------------------------------------------


@pytest.mark.parametrize("bad", [{"decay": 0.0}, {"decay": 1.5}, {"eta": -1.0}, {"write_every": 0}])
def test_bad_spec_rejected(bad):
    with pytest.raises(ValueError):
        MemorySpec(enabled=True, **bad)


# -- revisit schedule -----------------------------------------------------


def test_revisit_schedule_gives_distinct_names_for_the_same_environment():
    """A repeated split must produce distinct task names, or the results dict
    and the accuracy matrix silently collapse the two visits into one."""
    from bdh_cl.cl.tasks import build_mnist_sequence
    from bdh_cl.data import mnist

    cache = Path(__file__).resolve().parents[2] / "data_cache" / "mnist"
    if not all((cache / n).exists() for n in mnist.FILES.values()):
        pytest.skip("MNIST IDX cache not present")
    tasks = build_mnist_sequence(
        cache, side=7, n_eval=16, batch_size=4,
        splits=((0, 1), (2, 3), (0, 1)),
    )
    names = [t.name for t in tasks]
    assert names == ["mnist_01", "mnist_23", "mnist_01#2"]
    # the two visits are the SAME environment: same classes, same eval set
    assert tasks[0].pool.classes == tasks[2].pool.classes
    assert [e.prompt for e in tasks[0]._eval] == [e.prompt for e in tasks[2]._eval]
