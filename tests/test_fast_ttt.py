from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from tests.support import parameter_count, tensor_count
from ttt_svcbench_qwen.config import load_config
from ttt_svcbench_qwen.fast_ttt import (
    FastMemoryState,
    FastTTTAdapter,
    build_fast_ttt_adapter,
    deferred_fast_vjp_loss,
    make_query_proxy_fast_state,
)
from ttt_svcbench_qwen.qwen_adapter import QwenVideoFeatureBoundary

MEMORY_INTERFACE_PARAMETER_COUNT = 1_231_298


def make_adapter(*, dtype: torch.dtype = torch.float32) -> FastTTTAdapter:
    torch.manual_seed(11)
    return build_fast_ttt_adapter(load_config()).to(dtype=dtype)


def storage_pointer(tensor: Tensor) -> int:
    return int(tensor.untyped_storage().data_ptr())


def _written_state(state: FastMemoryState, magnitude: float = 0.05) -> FastMemoryState:
    memory = state.m.detach().clone()
    with torch.no_grad():
        memory[0, 0] = magnitude
    return replace(
        state,
        m=memory.requires_grad_(True),
        write_version=1,
        write_count=1,
    )


def test_structure_parameter_groups_and_checkpoint_keys_are_exact_on_meta() -> None:
    with torch.device("meta"):
        adapter = build_fast_ttt_adapter(load_config())

    assert isinstance(adapter.rms_norm, nn.RMSNorm)
    assert adapter.rms_norm.eps == 1.0e-6
    assert adapter.p_in.in_features == 4096
    assert adapter.p_in.out_features == 768
    assert adapter.p_out.in_features == 768
    assert adapter.p_out.out_features == 4096
    assert adapter.p_context.in_features == 512
    assert adapter.p_context.out_features == 768
    assert adapter.w0_1.shape == adapter.w0_2.shape == (768, 768)
    assert adapter.memory_key_probe.weight.shape == (768, 768)
    assert adapter.memory_value_projection.weight.shape == (768, 768)
    assert adapter.memory_eta_gate_hidden.in_features == 769
    assert adapter.memory_eta_gate_output.out_features == 1
    assert adapter.memory_alpha.shape == (768,)
    assert adapter.memory_beta_raw.shape == ()
    assert parameter_count(adapter) == 7_874_048 + MEMORY_INTERFACE_PARAMETER_COUNT
    assert tensor_count(adapter.collect_slow_parameters()) == 6_300_416
    assert (
        tensor_count(adapter.collect_associative_parameters())
        == 393_984 + MEMORY_INTERFACE_PARAMETER_COUNT
    )
    assert (
        sum(parameter.numel() for parameter in adapter.collect_meta_fast_parameters()) == 1_179_648
    )
    assert set(adapter.state_dict()) == {
        "rms_norm.weight",
        "p_in.weight",
        "p_in.bias",
        "p_context.weight",
        "p_context.bias",
        "w0_1",
        "w0_2",
        "p_out.weight",
        "p_out.bias",
        "memory_key_probe.weight",
        "memory_key_probe.bias",
        "memory_value_projection.weight",
        "memory_value_projection.bias",
        "memory_eta_gate_hidden.weight",
        "memory_eta_gate_hidden.bias",
        "memory_eta_gate_output.weight",
        "memory_eta_gate_output.bias",
        "memory_alpha",
        "memory_beta_raw",
        "memory_contract_version",
    }


def test_memory_gate_initialization_matches_the_frozen_schema() -> None:
    config = load_config()
    adapter = make_adapter()
    # The derivation below is only the *intended* eta when the gate's data term
    # vanishes, so pin the thing that makes it true: a zero output projection
    # forces every logit to equal the bias, on every seed and at any slot scale.
    # That is what keeps sum(eta) off the eta_chunk_budget annihilation point.
    assert not adapter.memory_eta_gate_output.weight.detach().any()
    eta_at_zero_input = config.fast_memory.eta_max_per_slot * torch.sigmoid(
        adapter.memory_eta_gate_output.bias.detach()
    )
    assert float(eta_at_zero_input) == pytest.approx(config.fast_memory.eta_gate_init, rel=1.0e-5)
    beta = config.fast_memory.forget_beta_max * torch.sigmoid(adapter.memory_beta_raw.detach())
    assert float(beta) == pytest.approx(config.fast_memory.forget_beta_init, rel=1.0e-5)
    assert torch.equal(
        adapter.memory_alpha.detach(),
        torch.full((768,), config.fast_memory.read_gate_init),
    )
    assert int(adapter.memory_contract_version.item()) == 4


def test_demo_forward_preserves_shape_dtype_device_and_stays_a_small_residual() -> None:
    adapter = make_adapter().eval()
    visual = torch.randn(1, 392, 4096)
    hook_outputs: list[Tensor] = []
    handle = adapter.register_forward_hook(
        lambda _module, _inputs, output: hook_outputs.append(output)
    )

    with torch.no_grad():
        output = adapter(visual)
    handle.remove()

    assert output.shape == visual.shape == (1, 392, 4096)
    assert output.dtype == visual.dtype
    assert output.device == visual.device
    assert torch.isfinite(output).all()
    assert hook_outputs == [output]
    residual_norm = torch.linalg.vector_norm(output - visual).item()
    assert residual_norm / torch.linalg.vector_norm(visual).item() < 0.25


def test_forward_matches_frozen_formula_exactly() -> None:
    adapter = make_adapter().eval()
    visual = torch.randn(1, 1, 4096)

    with torch.no_grad():
        expected = F.rms_norm(
            visual,
            (4096,),
            adapter.rms_norm.weight,
            adapter.rms_norm.eps,
        )
        expected = adapter.p_in(expected)
        expected = F.linear(expected, adapter.w0_1)
        expected = F.silu(expected)
        expected = F.linear(expected, adapter.w0_2)
        expected = visual + 0.1 * adapter.p_out(expected)
        actual = adapter(visual)

    assert torch.equal(actual, expected)


def test_zero_memory_forward_is_bitwise_identical_to_static_forward() -> None:
    """The A2-preservation regression: M=0 must change nothing, bit for bit."""

    adapter = make_adapter().eval()
    visual = torch.randn(2, 7, 4096)
    mask = torch.ones(2, 7, dtype=torch.bool)

    static_output = adapter(visual, mask)
    states = tuple(adapter.initialize_fast_state(differentiable=True) for _ in range(2))
    bound_output = adapter(visual, mask, fast_state=states)

    assert torch.equal(bound_output, static_output)
    online = tuple(adapter.initialize_fast_state() for _ in range(2))
    with adapter.use_fast_state(online):
        online_output = adapter(visual, mask)
    assert torch.equal(online_output, static_output)


def test_memory_keys_are_token_centered_and_split_from_the_core_input() -> None:
    """The memory key space removes the shared token mean; the W0 input keeps it.

    With no associative context bound the p_context term is exactly zero, so
    the memory keys must (a) average to ~zero over the valid tokens -- the
    0.46-0.50 anisotropy floor is gone -- while the uncentered projection does
    not, (b) reconstruct exactly from the module's own projections with the
    mean taken over valid tokens only, (c) fall back to the uncentered
    projection when a row has fewer than two valid tokens, and (d) leave the W0
    core path itself untouched, which is what keeps the M=0 forward bitwise.
    """

    torch.manual_seed(3)
    adapter = make_adapter().eval()
    visual = torch.randn(2, 9, 4096)
    mask = torch.tensor([[True] * 6 + [False] * 3, [True] + [False] * 8])
    state = tuple(adapter.initialize_fast_state() for _ in range(2))

    with torch.no_grad():
        static_output = adapter(visual, mask)
        bound_output = adapter(visual, mask, fast_state=state)
        keys = adapter.consume_associative_intermediates().keys
        base = F.rms_norm(visual, (4096,), adapter.rms_norm.weight, adapter.rms_norm.eps)
        base = adapter.p_in(base).float()

    scaled_mask = mask.unsqueeze(-1).to(dtype=base.dtype)
    counts = scaled_mask.sum(dim=1, keepdim=True)
    masked_mean = (base * scaled_mask).sum(dim=1, keepdim=True) / counts.clamp_min(1.0)

    # (a) valid-token mean of the memory keys vanishes; the raw projection's does not.
    assert float(keys[0, :6].mean(dim=0).norm()) < 1.0e-4
    assert float(base[0, :6].mean(dim=0).norm()) > 1.0e-2
    # (b) exact reconstruction, mean over valid tokens only (invalid rows shift too).
    torch.testing.assert_close(keys[0], base[0] - masked_mean[0], rtol=1.0e-5, atol=1.0e-5)
    # (c) a single-valid-token row passes through uncentered.
    torch.testing.assert_close(keys[1], base[1], rtol=1.0e-6, atol=1.0e-6)
    # (d) centering is a separate tensor: the W0 path is byte-for-byte unchanged.
    assert torch.equal(bound_output, static_output)


def test_readout_consumes_the_centered_memory_keys() -> None:
    """The read and write sides must share one key space.

    Reconstructs the bound forward from the static forward plus a readout built
    on the *centered* keys; if the readout bmm still consumed the uncentered W0
    input, this equality would fail by the mean term, and stored keys (written
    centered) would be probed by uncentered read queries -- the mismatch the
    split-tensor design exists to prevent.
    """

    torch.manual_seed(4)
    adapter = make_adapter().eval()
    visual = torch.randn(1, 5, 4096)
    mask = torch.ones(1, 5, dtype=torch.bool)
    state = adapter.initialize_fast_state()
    with torch.no_grad():
        state.m.copy_(torch.randn(768, 768) * 0.01)
        static_output = adapter(visual, mask)
        bound_output = adapter(visual, mask, fast_state=(state,))
        keys = adapter.consume_associative_intermediates().keys
        readout = adapter.memory_alpha.float() * (keys[0] @ state.m.detach().t())
        expected = static_output + 0.1 * F.linear(readout.unsqueeze(0), adapter.p_out.weight, None)
    torch.testing.assert_close(bound_output, expected, rtol=1.0e-5, atol=1.0e-6)
    assert not torch.equal(bound_output, static_output)


def test_runtime_batch_uses_one_independent_state_per_row_and_preserves_padding() -> None:
    """Per-row state isolation, read off the returned states and outputs.

    Both rows see the same visual row, so any difference in the output is
    attributable to the per-row memory alone: the zero-memory row must land
    bitwise on the static forward while the written row must not, and neither
    row's state object may be mutated by the read.
    """

    adapter = make_adapter().eval()
    first = adapter.initialize_fast_state()
    second = _written_state(adapter.initialize_fast_state(), magnitude=5.0)
    assert storage_pointer(first.m) != storage_pointer(second.m)
    row = torch.randn(1, 3, 4096)
    visual = row.repeat(2, 1, 1)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    counters = tuple(
        (state.write_version, state.write_count, state.skip_count) for state in (first, second)
    )
    values = tuple(state.m.detach().clone() for state in (first, second))

    static_output = adapter(visual, mask)
    output = adapter(visual, mask, fast_state=(first, second))

    # Padding is passed through untouched on both rows.
    assert torch.equal(output[~mask], visual[~mask])
    # Row 0 carries a zero memory, so it reproduces the static forward exactly;
    # row 1 carries a written memory and must diverge on its one valid token.
    assert torch.equal(output[0], static_output[0])
    assert not torch.equal(output[1, 0], static_output[1, 0])
    assert not torch.equal(output[0, 0], output[1, 0])
    # The read is non-destructive: neither memory value nor counters move.
    for state, counter, value in zip((first, second), counters, values, strict=True):
        assert (state.write_version, state.write_count, state.skip_count) == counter
        assert torch.equal(state.m.detach(), value)


def test_online_forward_gives_only_memory_and_input_gradients() -> None:
    adapter = make_adapter()
    state = _written_state(adapter.initialize_fast_state())
    visual = torch.randn(1, 2, 4096, requires_grad=True)
    state_value = state.m.detach().clone()
    state_storage = storage_pointer(state.m)
    state_counters = (state.write_version, state.write_count, state.skip_count)

    output = adapter(visual, fast_state=state)
    output.square().mean().backward()

    assert visual.grad is not None and torch.isfinite(visual.grad).all()
    for fast_parameter in state.fast_parameters:
        assert fast_parameter.grad is not None
        assert torch.isfinite(fast_parameter.grad).all()
        assert fast_parameter.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in adapter.parameters())
    assert (state.write_version, state.write_count, state.skip_count) == state_counters
    assert storage_pointer(state.m) == state_storage
    assert torch.equal(state.m.detach(), state_value)


def test_differentiable_state_preserves_outer_gradients_to_w0_and_slow_parameters() -> None:
    adapter = make_adapter()
    state = adapter.initialize_fast_state(differentiable=True)
    visual = torch.randn(1, 2, 4096, requires_grad=True)

    adapter(visual, fast_state=state).square().mean().backward()

    for parameter in (
        *adapter.collect_slow_parameters(),
        *adapter.collect_meta_fast_parameters(),
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert all(parameter.grad is not None for parameter in adapter.p_context.parameters())
    # The zero-init memory cannot be co-opted as static capacity: its readout is
    # exactly zero, so the read gate receives a zero (not missing) gradient.
    assert adapter.memory_alpha.grad is not None
    assert torch.count_nonzero(adapter.memory_alpha.grad) == 0
    assert state.m.grad is not None


def test_reset_and_video_initialization_are_zero_without_storage_sharing() -> None:
    adapter = make_adapter()
    first = adapter.initialize_fast_state()
    second = adapter.initialize_fast_state()
    assert storage_pointer(first.m) != storage_pointer(second.m)
    assert torch.count_nonzero(first.m.detach()) == 0
    assert torch.count_nonzero(second.m.detach()) == 0

    changed = _written_state(first)
    changed = replace(changed, write_version=3, write_count=3, skip_count=2)
    reset = adapter.reset_fast_state(changed)

    assert reset.write_version == reset.write_count == reset.skip_count == 0
    assert torch.count_nonzero(reset.m.detach()) == 0
    assert storage_pointer(reset.m) not in {
        storage_pointer(first.m),
        storage_pointer(second.m),
    }


def test_state_dict_roundtrip_saves_w0_and_never_the_transient_memory() -> None:
    source = make_adapter()
    state = _written_state(source.initialize_fast_state(), magnitude=5.0)
    checkpoint = {key: value.detach().clone() for key, value in source.state_dict().items()}
    target = make_adapter()
    target.load_state_dict(checkpoint)

    assert torch.equal(target.w0_1, source.w0_1)
    assert torch.equal(target.w0_2, source.w0_2)
    assert state.m.detach().abs().max() > 0
    assert all(key.split(".")[-1] != "m" and "active_fast" not in key for key in checkpoint)


def test_context_binding_integrates_with_p3_boundary_and_releases_after_errors() -> None:
    adapter = make_adapter().eval()
    assert adapter.p_out.bias is not None
    adapter.p_out.bias.requires_grad_(False)
    state = _written_state(adapter.initialize_fast_state())
    boundary = QwenVideoFeatureBoundary(load_config(), adapter, adapter_enabled=True)
    grid = torch.tensor([[1, 2, 2]], dtype=torch.int64)
    main = (torch.randn(1, 4096),)
    deepstack = [torch.randn(1, 4096) for _ in range(3)]
    outer_flags = tuple(parameter.requires_grad for parameter in adapter.parameters())

    with adapter.use_fast_state(state):
        assert all(not parameter.requires_grad for parameter in adapter.parameters())
        adapted, returned_deepstack = boundary.intercept_features(main, deepstack, grid)

    assert not torch.equal(adapted[0], main[0])
    assert returned_deepstack is deepstack
    assert all(key.split(".")[-1] != "m" for key in boundary.state_dict())
    assert tuple(parameter.requires_grad for parameter in adapter.parameters()) == outer_flags
    # The exception path must still release the binding and restore the flags.
    with pytest.raises(RuntimeError, match="sentinel"), adapter.use_fast_state(state):
        raise RuntimeError("sentinel")
    assert tuple(parameter.requires_grad for parameter in adapter.parameters()) == outer_flags
    # Release is observed on the output itself: outside the binding the written
    # memory is no longer read, so the same input must not produce the bound value.
    probe = main[0].unsqueeze(0)
    released_output = adapter(probe)
    with adapter.use_fast_state(state):
        bound_output = adapter(probe)
    assert not torch.equal(released_output, bound_output)
    assert torch.equal(released_output, adapter(probe))


@pytest.mark.parametrize("dtype", (torch.float64, torch.bfloat16))
def test_module_dtype_is_preserved_around_the_fp32_memory_core(dtype: torch.dtype) -> None:
    adapter = make_adapter(dtype=dtype)
    state = adapter.initialize_fast_state()
    visual = torch.randn(1, 1, 4096, dtype=dtype)

    output = adapter(visual, fast_state=state)

    assert output.dtype == dtype
    assert output.device == visual.device
    assert torch.isfinite(output).all()
    assert state.m.dtype == torch.float32


def test_small_memory_delta_changes_fp32_fast_core_prediction() -> None:
    adapter = make_adapter(dtype=torch.bfloat16)
    baseline = adapter.initialize_fast_state()
    visual = torch.randn(1, 1, 4096, dtype=torch.bfloat16)

    adapter(visual, fast_state=baseline)
    baseline_prediction = adapter.consume_associative_intermediates().predictions.detach()
    changed_memory = (baseline.m.detach() + 1.0e-3).requires_grad_(True)
    changed = replace(baseline, m=changed_memory)

    changed_output = adapter(visual, fast_state=changed)
    changed_prediction = adapter.consume_associative_intermediates().predictions.detach()

    assert baseline_prediction.dtype == changed_prediction.dtype == torch.float32
    assert not torch.equal(baseline_prediction, changed_prediction)
    # A nonzero memory must move the readout, hence the emitted embeddings too.
    assert changed_memory.detach().abs().max() > 0.0
    assert not torch.equal(changed_output, adapter(visual, fast_state=baseline))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is optional for local P5 checks")
def test_cuda_runtime_preserves_device() -> None:
    adapter = make_adapter().cuda()
    state = adapter.initialize_fast_state()
    visual = torch.randn(1, 1, 4096, device="cuda")

    output = adapter(visual, fast_state=state)

    assert output.device.type == "cuda"
    assert all(parameter.device.type == "cuda" for parameter in state.fast_parameters)


@pytest.mark.parametrize("query_count", (2, 4, 15))
def test_query_proxy_deferred_vjp_matches_one_shot_gradient(query_count: int) -> None:
    reference = make_adapter(dtype=torch.float32)
    streamed = make_adapter(dtype=torch.float32)
    streamed.load_state_dict(reference.state_dict())
    direct_reference = nn.Parameter(torch.tensor(0.125, dtype=torch.float32))
    direct_streamed = nn.Parameter(direct_reference.detach().clone())

    def updated_state(adapter: FastTTTAdapter) -> FastMemoryState:
        initial = adapter.initialize_fast_state(differentiable=True)
        # A differentiable write analog: the memory becomes a function of the
        # read gate so the deferred VJP has an outer parameter to reach.
        memory = initial.m + 0.01 * torch.tanh(adapter.memory_alpha).unsqueeze(0)
        return replace(initial, m=memory, write_version=1, write_count=1)

    reference_state = updated_state(reference)
    cases = tuple((0.5 + index / 7.0, -0.25 + index / 11.0) for index in range(query_count))
    reference_losses = tuple(
        (scale * reference_state.m[:2, :3] + direct_reference - target).square().mean()
        for scale, target in cases
    )
    torch.stack(reference_losses).mean().backward()

    streamed_state = updated_state(streamed)
    accumulated = tuple(
        torch.zeros_like(value, dtype=torch.float32) for value in streamed_state.fast_parameters
    )
    for scale, target in cases:
        proxy = make_query_proxy_fast_state(streamed_state)
        query_loss = (scale * proxy.m[:2, :3] + direct_streamed - target).square().mean() / float(
            query_count
        )
        query_loss.backward()
        for total, value in zip(accumulated, proxy.fast_parameters, strict=True):
            assert value.grad is not None
            total.add_(value.grad.detach())
    deferred_fast_vjp_loss((streamed_state,), accumulated).backward()

    assert reference.memory_alpha.grad is not None and streamed.memory_alpha.grad is not None
    assert direct_reference.grad is not None and direct_streamed.grad is not None
    assert torch.allclose(
        streamed.memory_alpha.grad,
        reference.memory_alpha.grad,
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    assert torch.allclose(direct_streamed.grad, direct_reference.grad, atol=1.0e-6, rtol=1.0e-6)


def test_memory_probe_parameters_partition_the_associative_group() -> None:
    adapter = build_fast_ttt_adapter(load_config())
    associative = {id(value) for value in adapter.collect_associative_parameters()}
    write = {id(value) for value in adapter.collect_memory_write_parameters()}
    read = {id(value) for value in adapter.collect_memory_read_parameters()}
    context = {id(adapter.p_context.weight), id(adapter.p_context.bias)}

    # The probes must exactly partition the pooled optimizer group, or a future
    # parameter would silently escape gradient observability.
    assert write | read | context == associative
    assert not write & read
    assert not (write | read) & context
    # memory_alpha gates the read path (alpha * K M^T) and would keep looking
    # healthy across a dead write path; memory_beta_raw is write-only.
    assert id(adapter.memory_alpha) in read
    assert id(adapter.memory_beta_raw) in write
    assert len(adapter.collect_memory_write_parameters()) == 9
