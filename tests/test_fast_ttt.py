from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ttt_svcbench_qwen.config import load_config
from ttt_svcbench_qwen.fast_ttt import (
    FastTTTAdapter,
    FastTTTForwardAudit,
    FastWeightsState,
    adapter_parameter_count,
    build_fast_ttt_adapter,
    collect_fast_parameters,
    online_parameter_count,
    slow_parameter_count,
)
from ttt_svcbench_qwen.qwen_adapter import QwenVideoFeatureBoundary


def make_adapter(*, dtype: torch.dtype = torch.float32) -> FastTTTAdapter:
    torch.manual_seed(11)
    return build_fast_ttt_adapter(load_config()).to(dtype=dtype)


def storage_pointer(tensor: Tensor) -> int:
    return int(tensor.untyped_storage().data_ptr())


def test_structure_parameter_groups_and_checkpoint_keys_are_exact_on_meta() -> None:
    with torch.device("meta"):
        adapter = build_fast_ttt_adapter(load_config())

    assert isinstance(adapter.rms_norm, nn.RMSNorm)
    assert adapter.rms_norm.eps == 1.0e-6
    assert adapter.p_in.in_features == 4096
    assert adapter.p_in.out_features == 768
    assert adapter.p_out.in_features == 768
    assert adapter.p_out.out_features == 4096
    assert adapter.p_in.bias is not None
    assert adapter.p_out.bias is not None
    assert adapter.w0_1.shape == adapter.w0_2.shape == (768, 768)
    assert set(adapter._modules) == {"rms_norm", "p_in", "p_out"}
    assert adapter_parameter_count(adapter) == 7_480_064
    assert slow_parameter_count(adapter) == 6_300_416
    assert (
        sum(parameter.numel() for parameter in adapter.collect_meta_fast_parameters())
        == 1_179_648
    )
    assert set(adapter.state_dict()) == {
        "rms_norm.weight",
        "p_in.weight",
        "p_in.bias",
        "w0_1",
        "w0_2",
        "p_out.weight",
        "p_out.bias",
    }


def test_demo_forward_preserves_shape_dtype_device_and_reports_true_residual_norm() -> None:
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
    audit = adapter.last_audit
    assert audit is not None
    assert audit.fast_versions == audit.update_counts == (0,)
    assert audit.valid_token_counts == (392,)
    assert audit.used_runtime_state is False
    actual_residual_norm = torch.linalg.vector_norm(output - visual).item()
    assert audit.residual_norms[0] == pytest.approx(actual_residual_norm, rel=1.0e-5)
    assert actual_residual_norm / torch.linalg.vector_norm(visual).item() < 0.25


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


def test_runtime_batch_uses_one_independent_state_per_row_and_preserves_padding() -> None:
    adapter = make_adapter().eval()
    first = adapter.initialize_fast_state()
    second = adapter.initialize_fast_state()
    with torch.no_grad():
        second.w_t_1.add_(0.05)
    second = replace(second, fast_version=2, update_count=2)
    row = torch.randn(1, 3, 4096)
    visual = row.repeat(2, 1, 1)
    mask = torch.tensor([[True, True, False], [True, False, False]])

    output = adapter(visual, mask, fast_state=(first, second))

    assert torch.equal(output[~mask], visual[~mask])
    assert not torch.equal(output[0, 0], output[1, 0])
    audit = adapter.last_audit
    assert audit is not None
    assert audit.fast_versions == audit.update_counts == (0, 2)
    assert audit.valid_token_counts == (2, 1)
    assert audit.used_runtime_state is True
    for batch_index in range(2):
        delta = output[batch_index][mask[batch_index]] - visual[batch_index][mask[batch_index]]
        assert audit.residual_norms[batch_index] == pytest.approx(
            torch.linalg.vector_norm(delta).item(),
            rel=1.0e-5,
        )

    with pytest.raises(ValueError, match="one runtime state per batch item"):
        adapter(visual, mask, fast_state=first)
    with pytest.raises(ValueError, match="one runtime state per batch item"):
        adapter(visual, mask, fast_state=(first,))
    with pytest.raises(ValueError, match="must not share W_t storage"):
        adapter(visual, mask, fast_state=(first, first))
    shared = FastWeightsState(
        second.w0_1,
        second.w0_2,
        first.w_t_1,
        second.w_t_2,
        0,
        0,
        0,
    )
    with pytest.raises(ValueError, match="must not share W_t storage"):
        adapter(visual, mask, fast_state=(first, shared))
    differentiable = adapter.initialize_fast_state(differentiable=True)
    with pytest.raises(ValueError, match="cannot mix"):
        adapter(visual, mask, fast_state=(first, differentiable))


def test_online_forward_gives_only_w_t_and_input_gradients() -> None:
    adapter = make_adapter()
    state = adapter.initialize_fast_state()
    visual = torch.randn(1, 2, 4096, requires_grad=True)
    state_tensors = (state.w0_1, state.w0_2, state.w_t_1, state.w_t_2)
    state_values = tuple(tensor.detach().clone() for tensor in state_tensors)
    state_storage = tuple(storage_pointer(tensor) for tensor in state_tensors)
    state_counters = (state.fast_version, state.update_count, state.skip_count)

    output = adapter(visual, fast_state=state)
    output.square().mean().backward()

    assert visual.grad is not None and torch.isfinite(visual.grad).all()
    for fast_parameter in collect_fast_parameters(state):
        assert fast_parameter.grad is not None
        assert torch.isfinite(fast_parameter.grad).all()
        assert fast_parameter.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in adapter.parameters())
    assert (state.fast_version, state.update_count, state.skip_count) == state_counters
    assert tuple(storage_pointer(tensor) for tensor in state_tensors) == state_storage
    for tensor, expected in zip(state_tensors, state_values, strict=True):
        assert torch.equal(tensor, expected)


def test_differentiable_state_preserves_outer_gradients_to_w0_and_slow_parameters() -> None:
    adapter = make_adapter()
    state = adapter.initialize_fast_state(differentiable=True)
    visual = torch.randn(1, 2, 4096, requires_grad=True)

    adapter(visual, fast_state=state).square().mean().backward()

    assert state.w_t_1.is_leaf is False
    assert state.w_t_2.is_leaf is False
    for parameter in adapter.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_reset_and_video_initialization_clone_current_w0_without_storage_sharing() -> None:
    adapter = make_adapter()
    first = adapter.initialize_fast_state()
    second = adapter.initialize_fast_state()
    all_runtime = (
        first.w0_1,
        first.w0_2,
        first.w_t_1,
        first.w_t_2,
        second.w0_1,
        second.w0_2,
        second.w_t_1,
        second.w_t_2,
    )
    assert len({storage_pointer(tensor) for tensor in all_runtime}) == len(all_runtime)
    assert torch.equal(first.w_t_1, adapter.w0_1)
    assert torch.equal(first.w_t_2, adapter.w0_2)

    with torch.no_grad():
        first.w_t_1.add_(1.0)
        adapter.w0_1.add_(0.25)
    changed = replace(first, fast_version=3, update_count=3, skip_count=2)
    reset = adapter.reset_fast_state(changed)

    assert reset.fast_version == reset.update_count == reset.skip_count == 0
    assert torch.equal(reset.w_t_1, adapter.w0_1)
    assert torch.equal(reset.w_t_2, adapter.w0_2)
    assert storage_pointer(reset.w_t_1) not in {
        storage_pointer(first.w_t_1),
        storage_pointer(second.w_t_1),
        storage_pointer(adapter.w0_1),
    }


def test_parameter_collection_is_stable_exact_and_rejects_boundary_drift() -> None:
    adapter = make_adapter()
    state = adapter.initialize_fast_state()
    groups = adapter.parameter_groups(state)

    assert groups.meta_fast == (adapter.w0_1, adapter.w0_2)
    assert groups.online_fast[0] is state.w_t_1
    assert groups.online_fast[1] is state.w_t_2
    assert groups.slow == adapter.collect_slow_parameters()
    assert online_parameter_count(state) == 2 * 768 * 768 == 1_179_648
    assert slow_parameter_count(adapter) == 6_300_416
    assert not ({id(parameter) for parameter in groups.online_fast} & {id(p) for p in groups.slow})
    adapter.assert_online_parameter_boundary(groups.online_fast, state)

    with pytest.raises(ValueError, match="stable order"):
        adapter.assert_online_parameter_boundary(groups.online_fast[::-1], state)
    with pytest.raises(ValueError, match="exactly"):
        adapter.assert_online_parameter_boundary((*groups.online_fast, adapter.p_in.weight), state)
    with pytest.raises(ValueError, match="exactly"):
        adapter.assert_online_parameter_boundary(groups.meta_fast, state)


def test_state_dict_roundtrip_saves_w0_and_never_transient_w_t() -> None:
    source = make_adapter()
    state = source.initialize_fast_state()
    with torch.no_grad():
        state.w_t_1.add_(5.0)
    checkpoint = {key: value.detach().clone() for key, value in source.state_dict().items()}
    target = make_adapter()
    target.load_state_dict(checkpoint)

    assert torch.equal(target.w0_1, source.w0_1)
    assert torch.equal(target.w0_2, source.w0_2)
    assert not torch.equal(target.w0_1, state.w_t_1)
    assert all("w_t" not in key and "active_fast" not in key for key in checkpoint)


def test_context_binding_integrates_with_p3_boundary_and_cleans_up_after_errors() -> None:
    adapter = make_adapter().eval()
    assert adapter.p_out.bias is not None
    adapter.p_out.bias.requires_grad_(False)
    state = adapter.initialize_fast_state()
    with torch.no_grad():
        state.w_t_2.mul_(0.5)
    boundary = QwenVideoFeatureBoundary(load_config(), adapter, adapter_enabled=True)
    grid = torch.tensor([[1, 2, 2]], dtype=torch.int64)
    main = (torch.randn(1, 4096),)
    deepstack = [torch.randn(1, 4096) for _ in range(3)]
    outer_flags = tuple(parameter.requires_grad for parameter in adapter.parameters())

    with adapter.use_fast_state(state):
        assert all(not parameter.requires_grad for parameter in adapter.parameters())
        adapter.assert_online_freeze((state,))
        adapted, returned_deepstack = boundary.intercept_features(main, deepstack, grid)
        assert adapter.last_audit is not None
        assert adapter.last_audit.used_runtime_state is True
        with pytest.raises(RuntimeError, match="not re-entrant"), adapter.use_fast_state(state):
            pass

    assert not torch.equal(adapted[0], main[0])
    assert returned_deepstack is deepstack
    assert all("w_t" not in key for key in boundary.state_dict())
    assert tuple(parameter.requires_grad for parameter in adapter.parameters()) == outer_flags
    with pytest.raises(RuntimeError, match="sentinel"), adapter.use_fast_state(state):
        raise RuntimeError("sentinel")
    assert tuple(parameter.requires_grad for parameter in adapter.parameters()) == outer_flags
    adapter(main[0].unsqueeze(0))
    assert adapter.last_audit is not None
    assert adapter.last_audit.used_runtime_state is False


def test_online_binding_rejects_stale_slow_grad_and_differentiable_binding_stays_outer() -> None:
    adapter = make_adapter()
    online_state = adapter.initialize_fast_state()
    outer_flags = tuple(parameter.requires_grad for parameter in adapter.parameters())
    adapter.p_in.weight.grad = torch.ones_like(adapter.p_in.weight)
    with pytest.raises(
        ValueError, match="stale module gradients"
    ), adapter.use_fast_state(online_state):
        pass
    assert tuple(parameter.requires_grad for parameter in adapter.parameters()) == outer_flags
    adapter.zero_grad(set_to_none=True)

    differentiable_state = adapter.initialize_fast_state(differentiable=True)
    with adapter.use_fast_state(differentiable_state):
        assert all(parameter.requires_grad for parameter in adapter.parameters())
        output = adapter(torch.randn(1, 1, 4096))
    output.square().mean().backward()
    assert all(parameter.grad is not None for parameter in adapter.parameters())


def test_float64_is_preserved_and_stale_state_after_module_move_fails() -> None:
    adapter = make_adapter()
    stale = adapter.initialize_fast_state()
    adapter = adapter.to(dtype=torch.float64)
    visual = torch.randn(1, 1, 4096, dtype=torch.float64)

    output = adapter(visual)

    assert output.dtype == torch.float64
    assert output.device == visual.device
    with pytest.raises(ValueError, match="module dtype/device"):
        adapter(visual, fast_state=stale)


def test_bfloat16_online_runtime_preserves_dtype() -> None:
    adapter = make_adapter(dtype=torch.bfloat16)
    state = adapter.initialize_fast_state()
    visual = torch.randn(1, 1, 4096, dtype=torch.bfloat16)

    output = adapter(visual, fast_state=state)

    assert output.dtype == torch.bfloat16
    assert output.device == visual.device
    assert torch.isfinite(output).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is optional for local P5 checks")
def test_cuda_runtime_preserves_device() -> None:
    adapter = make_adapter().cuda()
    state = adapter.initialize_fast_state()
    visual = torch.randn(1, 1, 4096, device="cuda")

    output = adapter(visual, fast_state=state)

    assert output.device.type == "cuda"
    assert all(parameter.device.type == "cuda" for parameter in collect_fast_parameters(state))


@pytest.mark.parametrize(
    ("visual", "mask", "reason"),
    [
        (torch.zeros(1, 4096), None, "non-empty"),
        (torch.zeros(1, 1, 4096, dtype=torch.int64), None, "floating"),
        (torch.full((1, 1, 4096), torch.nan), None, "finite"),
        (torch.full((1, 1, 4096), torch.inf), None, "finite"),
        (torch.zeros(1, 1, 4096, dtype=torch.float64), None, "share dtype/device"),
        (torch.zeros(1, 2, 4096), torch.ones(1, 2), "bool"),
        (torch.zeros(1, 2, 4096), torch.ones(1, 1, dtype=torch.bool), "bool"),
        (
            torch.zeros(1, 2, 4096),
            torch.ones(1, 2, dtype=torch.bool, device="meta"),
            "share a device",
        ),
        (torch.zeros(1, 2, 4096), torch.zeros(1, 2, dtype=torch.bool), "at least one"),
    ],
)
def test_forward_rejects_invalid_inputs_and_masks(
    visual: Tensor,
    mask: Tensor | None,
    reason: str,
) -> None:
    adapter = make_adapter()
    with pytest.raises((TypeError, ValueError), match=reason):
        adapter(visual, mask)


def test_fast_state_rejects_alias_nonfinite_nonleaf_and_invalid_metadata() -> None:
    w0_1 = torch.zeros(768, 768)
    w0_2 = torch.ones(768, 768)
    w_t_1 = w0_1.clone().requires_grad_(True)
    w_t_2 = w0_2.clone().requires_grad_(True)

    with pytest.raises(ValueError, match="distinct storage"):
        FastWeightsState(w0_1, w0_2, w0_1.view_as(w0_1), w_t_2, 0, 0, 0)
    bad = w_t_1.detach().clone().requires_grad_(True)
    with torch.no_grad():
        bad[0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        FastWeightsState(w0_1, w0_2, bad, w_t_2, 0, 0, 0)
    bad_inf = w_t_1.detach().clone().requires_grad_(True)
    with torch.no_grad():
        bad_inf[0, 0] = torch.inf
    with pytest.raises(ValueError, match="finite"):
        FastWeightsState(w0_1, w0_2, bad_inf, w_t_2, 0, 0, 0)
    nonleaf = w_t_1 + 1.0
    with pytest.raises(ValueError, match="leaf"):
        FastWeightsState(w0_1, w0_2, nonleaf, w_t_2, 0, 0, 0)
    with pytest.raises(TypeError, match="exact integers"):
        FastWeightsState(w0_1, w0_2, w_t_1, w_t_2, True, 1, 0)
    with pytest.raises(ValueError, match="accepted updates"):
        FastWeightsState(w0_1, w0_2, w_t_1, w_t_2, 2, 1, 0)
    with pytest.raises(TypeError, match="differentiable"):
        FastWeightsState(w0_1, w0_2, w_t_1, w_t_2, 0, 0, 0, 1)  # type: ignore[arg-type]

    adapter = make_adapter()
    state = adapter.initialize_fast_state()
    visual = torch.zeros(1, 1, 4096)
    with pytest.raises(TypeError, match="only FastWeightsState"):
        adapter(visual, fast_state=(object(),))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="reset mode"):
        adapter.reset_fast_state(state, differentiable=1)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="exact integers"):
        FastTTTForwardAudit(
            (True,),
            (0,),
            (1,),
            False,
            (0.0,),
            (0.0,),
            (0.0,),
            (0.0,),
        )
    with pytest.raises(TypeError, match="used_runtime_state"):
        FastTTTForwardAudit(
            (0,),
            (0,),
            (1,),
            1,  # type: ignore[arg-type]
            (0.0,),
            (0.0,),
            (0.0,),
            (0.0,),
        )


def test_builder_requires_validated_project_config() -> None:
    with pytest.raises(ValueError, match="validated ProjectConfig"):
        build_fast_ttt_adapter()
