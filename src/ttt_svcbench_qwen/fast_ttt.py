"""Implement the Fast TTT Adapter and its explicit per-video weight boundary.

Inputs: Main Merger embeddings, an optional padding mask, and explicit per-video W_t state.
Outputs: shape-preserving adapted embeddings plus detached numerical audit metadata.
Forbidden: optimizer steps, hidden W_t registration, State Bank mutation, query routing, gates, or
online gradients into W0/RMSNorm/P_in/P_out.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ttt_svcbench_qwen.config import FastTTTConfig, ProjectConfig


@dataclass(frozen=True, slots=True)
class FastWeightsState:
    """One video's W0 snapshot and current W_t tensors; never register this on an nn.Module."""

    w0_1: Tensor
    w0_2: Tensor
    w_t_1: Tensor
    w_t_2: Tensor
    fast_version: int
    update_count: int
    skip_count: int
    differentiable: bool = False

    def __post_init__(self) -> None:
        matrices = (self.w0_1, self.w0_2, self.w_t_1, self.w_t_2)
        for matrix in matrices:
            if matrix.shape != (768, 768) or not torch.is_floating_point(matrix):
                raise ValueError("all fast matrices must be floating [768, 768]")
            if matrix.dtype != self.w0_1.dtype or matrix.device != self.w0_1.device:
                raise ValueError("all fast matrices must share dtype and device")
            if matrix.device.type != "meta" and not bool(torch.isfinite(matrix).all()):
                raise ValueError("all fast matrices must be finite")
        for left_index, left in enumerate(matrices):
            for right in matrices[left_index + 1 :]:
                if _shares_storage(left, right):
                    raise ValueError("W0 and W_t matrices must use distinct storage")
        if not self.w_t_1.requires_grad or not self.w_t_2.requires_grad:
            raise ValueError("current fast matrices must require gradients")
        if type(self.differentiable) is not bool:
            raise TypeError("differentiable must be a bool")
        if not self.differentiable and (not self.w_t_1.is_leaf or not self.w_t_2.is_leaf):
            raise ValueError("non-differentiable runtime fast matrices must be leaf tensors")
        counters = (self.fast_version, self.update_count, self.skip_count)
        if any(type(counter) is not int for counter in counters):
            raise TypeError("fast runtime counters must be exact integers")
        if min(counters) < 0:
            raise ValueError("fast runtime counters must be non-negative")
        if self.fast_version != self.update_count:
            raise ValueError("fast_version must equal the number of accepted updates")

    @property
    def fast_parameters(self) -> tuple[Tensor, Tensor]:
        return (self.w_t_1, self.w_t_2)


@dataclass(frozen=True, slots=True)
class OptimizerRuntimeState:
    optimizer_name: str
    learning_rate: float
    momentum: float
    weight_decay: float
    steps_per_chunk: int
    grad_clip_norm: float
    attempted_update_count: int
    last_skip_reason: str | None

    def __post_init__(self) -> None:
        fixed = (
            self.optimizer_name == "sgd"
            and self.learning_rate == 1.0e-4
            and self.momentum == 0.0
            and self.weight_decay == 0.0
            and self.steps_per_chunk == 1
            and self.grad_clip_norm == 1.0
        )
        if not fixed:
            raise ValueError("optimizer runtime must match the frozen single-step SGD contract")
        if self.attempted_update_count < 0:
            raise ValueError("attempted_update_count must be non-negative")


@dataclass(frozen=True, slots=True)
class FastParameterGroups:
    """Separate checkpointed meta-fast, checkpointed slow, and transient online tensors."""

    meta_fast: tuple[nn.Parameter, nn.Parameter]
    slow: tuple[nn.Parameter, ...]
    online_fast: tuple[Tensor, Tensor]


@dataclass(frozen=True, slots=True)
class FastTTTForwardAudit:
    fast_versions: tuple[int, ...]
    update_counts: tuple[int, ...]
    valid_token_counts: tuple[int, ...]
    used_runtime_state: bool
    w_t_1_norms: tuple[float, ...]
    w_t_2_norms: tuple[float, ...]
    input_norms: tuple[float, ...]
    residual_norms: tuple[float, ...]

    def __post_init__(self) -> None:
        if type(self.used_runtime_state) is not bool:
            raise TypeError("used_runtime_state must be a bool")
        lengths = {
            len(self.fast_versions),
            len(self.update_counts),
            len(self.valid_token_counts),
            len(self.w_t_1_norms),
            len(self.w_t_2_norms),
            len(self.input_norms),
            len(self.residual_norms),
        }
        if lengths != {len(self.fast_versions)} or not self.fast_versions:
            raise ValueError("Fast TTT audit fields must align to one non-empty batch")
        counters = (*self.fast_versions, *self.update_counts, *self.valid_token_counts)
        if any(type(counter) is not int for counter in counters):
            raise TypeError("Fast TTT audit counters must be exact integers")
        if min(counters) < 0:
            raise ValueError("Fast TTT audit counters must be non-negative")
        if min(self.valid_token_counts) == 0:
            raise ValueError("Fast TTT audit requires at least one valid token per batch item")
        if self.fast_versions != self.update_counts:
            raise ValueError("Fast TTT audit version must equal accepted update count")
        values = (
            *self.w_t_1_norms,
            *self.w_t_2_norms,
            *self.input_norms,
            *self.residual_norms,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("Fast TTT audit norms must be finite and non-negative")


class FastTTTAdapter(nn.Module):  # type: ignore[misc]
    """4096→768→768→4096 residual Adapter with external per-video W_t tensors."""

    def __init__(self, config: FastTTTConfig) -> None:
        super().__init__()
        if config.fast_bias:
            raise ValueError("Fast TTT matrices must not use bias")
        if not config.slow_projection_bias:
            raise ValueError("P5 parameter budget requires bias on the slow projections")
        if config.fast_initialization != "xavier_uniform":
            raise ValueError("P5 requires xavier_uniform W0 initialization")
        self.config = config
        self.input_dim = config.input_dim
        self.bottleneck_dim = config.bottleneck_dim
        self.output_dim = config.output_dim
        self.residual_scale = config.residual_scale
        self.rms_norm = nn.RMSNorm(config.input_dim, eps=config.rms_norm_eps)
        self.p_in = nn.Linear(
            config.input_dim,
            config.bottleneck_dim,
            bias=config.slow_projection_bias,
        )
        self.w0_1 = nn.Parameter(torch.empty(config.bottleneck_dim, config.bottleneck_dim))
        self.w0_2 = nn.Parameter(torch.empty(config.bottleneck_dim, config.bottleneck_dim))
        self.p_out = nn.Linear(
            config.bottleneck_dim,
            config.output_dim,
            bias=config.slow_projection_bias,
        )
        nn.init.xavier_uniform_(self.w0_1)
        nn.init.xavier_uniform_(self.w0_2)
        self._active_fast_states: tuple[FastWeightsState, ...] | None = None
        self.last_audit: FastTTTForwardAudit | None = None

    def forward(
        self,
        visual_embeddings: Tensor,
        valid_mask: Tensor | None = None,
        metadata: object | None = None,
        *,
        fast_state: FastWeightsState | Sequence[FastWeightsState] | None = None,
    ) -> Tensor:
        """Use W0 for outer training, or detached slow parameters plus explicit W_t online."""

        del metadata
        self.last_audit = None
        self._validate_input(visual_embeddings)
        mask = self._normalize_valid_mask(visual_embeddings, valid_mask)
        if fast_state is not None and self._active_fast_states is not None:
            raise RuntimeError(
                "pass fast_state explicitly or bind one with use_fast_state(), not both"
            )
        raw_runtime_states = fast_state if fast_state is not None else self._active_fast_states
        runtime_states = self._normalize_runtime_states(
            raw_runtime_states,
            visual_embeddings.shape[0],
        )
        if runtime_states is None:
            w_t_1, w_t_2 = self.w0_1, self.w0_2
            detach_slow = False
            fast_versions = (0,) * visual_embeddings.shape[0]
            update_counts = fast_versions
        else:
            for state in runtime_states:
                self._validate_state_for_input(state, visual_embeddings)
            differentiable_modes = {state.differentiable for state in runtime_states}
            if len(differentiable_modes) != 1:
                raise ValueError("one Fast TTT batch cannot mix differentiable and online states")
            w_t_1 = torch.stack([state.w_t_1 for state in runtime_states])
            w_t_2 = torch.stack([state.w_t_2 for state in runtime_states])
            detach_slow = not runtime_states[0].differentiable
            fast_versions = tuple(state.fast_version for state in runtime_states)
            update_counts = tuple(state.update_count for state in runtime_states)

        rms_weight = self._online_value(self.rms_norm.weight, detach_slow)
        p_in_weight = self._online_value(self.p_in.weight, detach_slow)
        p_in_bias = self._online_value(self.p_in.bias, detach_slow)
        p_out_weight = self._online_value(self.p_out.weight, detach_slow)
        p_out_bias = self._online_value(self.p_out.bias, detach_slow)
        normalized = F.rms_norm(
            visual_embeddings,
            (self.input_dim,),
            rms_weight,
            self.config.rms_norm_eps,
        )
        projected = F.linear(normalized, p_in_weight, p_in_bias)
        if runtime_states is None:
            hidden = F.linear(projected, w_t_1, None)
        else:
            hidden = torch.bmm(projected, w_t_1.transpose(1, 2))
        hidden = F.silu(hidden)
        if runtime_states is None:
            hidden = F.linear(hidden, w_t_2, None)
        else:
            hidden = torch.bmm(hidden, w_t_2.transpose(1, 2))
        residual = F.linear(hidden, p_out_weight, p_out_bias)
        residual = residual.masked_fill(~mask.unsqueeze(-1), 0.0)
        scaled_residual = self.residual_scale * residual
        output = visual_embeddings + scaled_residual
        if not bool(torch.isfinite(output).all()):
            raise ValueError("Fast TTT output must be finite")
        if runtime_states is None:
            w_t_1_norms = (_detached_norm(w_t_1),) * visual_embeddings.shape[0]
            w_t_2_norms = (_detached_norm(w_t_2),) * visual_embeddings.shape[0]
        else:
            w_t_1_norms = tuple(_detached_norm(state.w_t_1) for state in runtime_states)
            w_t_2_norms = tuple(_detached_norm(state.w_t_2) for state in runtime_states)
        self.last_audit = FastTTTForwardAudit(
            fast_versions=fast_versions,
            update_counts=update_counts,
            valid_token_counts=tuple(int(row.sum().item()) for row in mask),
            used_runtime_state=runtime_states is not None,
            w_t_1_norms=w_t_1_norms,
            w_t_2_norms=w_t_2_norms,
            input_norms=tuple(
                _detached_norm(visual_embeddings[row][mask[row]])
                for row in range(visual_embeddings.shape[0])
            ),
            residual_norms=tuple(
                _detached_norm(scaled_residual[row][mask[row]])
                for row in range(visual_embeddings.shape[0])
            ),
        )
        return output

    def initialize_fast_state(self, *, differentiable: bool = False) -> FastWeightsState:
        """Clone checkpointed W0 into storage-independent per-video W_t tensors."""

        if differentiable:
            w0_1: Tensor = self.w0_1
            w0_2: Tensor = self.w0_2
            w_t_1 = self.w0_1.clone()
            w_t_2 = self.w0_2.clone()
        else:
            w0_1 = self.w0_1.detach().clone()
            w0_2 = self.w0_2.detach().clone()
            w_t_1 = w0_1.clone().requires_grad_(True)
            w_t_2 = w0_2.clone().requires_grad_(True)
        state = FastWeightsState(
            w0_1=w0_1,
            w0_2=w0_2,
            w_t_1=w_t_1,
            w_t_2=w_t_2,
            fast_version=0,
            update_count=0,
            skip_count=0,
            differentiable=differentiable,
        )
        self.assert_online_parameter_boundary(state.fast_parameters, state)
        return state

    def reset_fast_state(
        self,
        state: FastWeightsState | None = None,
        *,
        differentiable: bool | None = None,
    ) -> FastWeightsState:
        """Reset counters and clone the current meta-learned W0 for a fresh video episode."""

        if differentiable is not None and type(differentiable) is not bool:
            raise TypeError("differentiable reset mode must be a bool")
        mode = (
            state.differentiable
            if state is not None and differentiable is None
            else differentiable
        )
        return self.initialize_fast_state(differentiable=bool(mode))

    @contextmanager
    def use_fast_state(
        self,
        state: FastWeightsState | Sequence[FastWeightsState],
    ) -> Iterator[FastTTTAdapter]:
        """Temporarily bind W_t for the unchanged P3 Qwen adapter call signature."""

        if self._active_fast_states is not None:
            raise RuntimeError("Fast TTT runtime state binding is not re-entrant")
        states = (state,) if isinstance(state, FastWeightsState) else tuple(state)
        if not states:
            raise ValueError("Fast TTT runtime state binding cannot be empty")
        if not all(isinstance(item, FastWeightsState) for item in states):
            raise TypeError("Fast TTT bindings must contain only FastWeightsState values")
        _assert_batched_state_storage_isolated(states)
        for item in states:
            self._validate_state_for_module(item)
        if len({item.differentiable for item in states}) != 1:
            raise ValueError("one Fast TTT binding cannot mix differentiable and online states")
        freeze_module = not states[0].differentiable
        previous_requires_grad = tuple(parameter.requires_grad for parameter in self.parameters())
        if freeze_module:
            if any(parameter.grad is not None for parameter in self.parameters()):
                raise ValueError(
                    "clear stale module gradients before binding online Fast TTT state"
                )
            for parameter in self.parameters():
                parameter.requires_grad_(False)
        self._active_fast_states = states
        try:
            if freeze_module:
                self.assert_online_freeze(states)
            yield self
        finally:
            self._active_fast_states = None
            if freeze_module:
                for parameter, requires_grad in zip(
                    self.parameters(),
                    previous_requires_grad,
                    strict=True,
                ):
                    parameter.requires_grad_(requires_grad)

    def collect_meta_fast_parameters(self) -> tuple[nn.Parameter, nn.Parameter]:
        return (self.w0_1, self.w0_2)

    def collect_slow_parameters(self) -> tuple[nn.Parameter, ...]:
        bias_in = self.p_in.bias
        bias_out = self.p_out.bias
        if bias_in is None or bias_out is None:
            raise RuntimeError("P5 slow projection biases disappeared")
        return (
            self.rms_norm.weight,
            self.p_in.weight,
            bias_in,
            self.p_out.weight,
            bias_out,
        )

    def parameter_groups(self, state: FastWeightsState) -> FastParameterGroups:
        return FastParameterGroups(
            meta_fast=self.collect_meta_fast_parameters(),
            slow=self.collect_slow_parameters(),
            online_fast=collect_fast_parameters(state),
        )

    def assert_online_parameter_boundary(
        self,
        parameters: Iterable[Tensor],
        state: FastWeightsState,
    ) -> None:
        supplied = tuple(parameters)
        expected = collect_fast_parameters(state)
        if len(supplied) != 2 or any(
            actual is not required for actual, required in zip(supplied, expected, strict=True)
        ):
            raise ValueError("online parameters must be exactly (w_t_1, w_t_2) in stable order")
        if sum(parameter.numel() for parameter in supplied) != self.config.online_parameter_count:
            raise ValueError("online fast parameter count must equal 1,179,648")
        module_ids = {id(parameter) for parameter in self.parameters()}
        if any(id(parameter) in module_ids for parameter in supplied):
            raise ValueError("transient W_t tensors must not be registered module parameters")

    def assert_online_freeze(self, states: Sequence[FastWeightsState]) -> None:
        """Prove that an inference binding freezes every checkpointed parameter."""

        if not states or any(state.differentiable for state in states):
            raise ValueError("online freeze requires non-differentiable per-video states")
        for state in states:
            self.assert_online_parameter_boundary(state.fast_parameters, state)
        if any(parameter.requires_grad for parameter in self.parameters()):
            raise ValueError("all checkpointed Fast TTT parameters must be frozen online")
        if any(parameter.grad is not None for parameter in self.parameters()):
            raise ValueError("checkpointed Fast TTT parameters must not carry online gradients")

    def _validate_input(self, visual_embeddings: Tensor) -> None:
        if (
            visual_embeddings.ndim != 3
            or visual_embeddings.shape[0] <= 0
            or visual_embeddings.shape[1] <= 0
            or visual_embeddings.shape[2] != self.input_dim
        ):
            raise ValueError(f"visual_embeddings must be non-empty [B, N_v, {self.input_dim}]")
        if not torch.is_floating_point(visual_embeddings):
            raise TypeError("visual_embeddings must use a floating dtype")
        if not bool(torch.isfinite(visual_embeddings).all()):
            raise ValueError("visual_embeddings must be finite")
        for parameter in self.parameters():
            if (
                parameter.dtype != visual_embeddings.dtype
                or parameter.device != visual_embeddings.device
            ):
                raise ValueError("Fast TTT module and visual embeddings must share dtype/device")

    def _normalize_valid_mask(
        self,
        visual_embeddings: Tensor,
        valid_mask: Tensor | None,
    ) -> Tensor:
        if valid_mask is None:
            return torch.ones(
                visual_embeddings.shape[:2],
                dtype=torch.bool,
                device=visual_embeddings.device,
            )
        if valid_mask.shape != visual_embeddings.shape[:2] or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be bool [B, N_v]")
        if valid_mask.device != visual_embeddings.device:
            raise ValueError("valid_mask and visual embeddings must share a device")
        if bool(torch.any(valid_mask.sum(dim=1) == 0)):
            raise ValueError("every Fast TTT batch item must contain at least one valid token")
        return valid_mask

    def _validate_state_for_module(self, state: FastWeightsState) -> None:
        if state.w_t_1.dtype != self.w0_1.dtype or state.w_t_1.device != self.w0_1.device:
            raise ValueError("Fast TTT runtime state must share module dtype/device")
        self.assert_online_parameter_boundary(state.fast_parameters, state)

    def _validate_state_for_input(
        self,
        state: FastWeightsState,
        visual_embeddings: Tensor,
    ) -> None:
        self._validate_state_for_module(state)
        if (
            state.w_t_1.dtype != visual_embeddings.dtype
            or state.w_t_1.device != visual_embeddings.device
        ):
            raise ValueError("Fast TTT runtime state must share input dtype/device")

    def _normalize_runtime_states(
        self,
        state: FastWeightsState | Sequence[FastWeightsState] | None,
        batch_size: int,
    ) -> tuple[FastWeightsState, ...] | None:
        if state is None:
            return None
        states = (state,) if isinstance(state, FastWeightsState) else tuple(state)
        if not all(isinstance(item, FastWeightsState) for item in states):
            raise TypeError("Fast TTT state sequences must contain only FastWeightsState values")
        if len(states) != batch_size:
            raise ValueError("Fast TTT requires exactly one runtime state per batch item")
        _assert_batched_state_storage_isolated(states)
        return states

    @staticmethod
    def _online_value(value: Tensor | None, detach: bool) -> Tensor | None:
        if value is None:
            return None
        return value.detach() if detach else value


def collect_fast_parameters(state: FastWeightsState) -> tuple[Tensor, Tensor]:
    """Return only W_t^(1), W_t^(2), in stable formula order."""

    return state.fast_parameters


def adapter_parameter_count(module: FastTTTAdapter) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def slow_parameter_count(module: FastTTTAdapter) -> int:
    return sum(parameter.numel() for parameter in module.collect_slow_parameters())


def online_parameter_count(state: FastWeightsState) -> int:
    return sum(parameter.numel() for parameter in collect_fast_parameters(state))


def build_fast_ttt_adapter(config: ProjectConfig | None = None) -> FastTTTAdapter:
    if config is None:
        raise ValueError("build_fast_ttt_adapter requires a validated ProjectConfig")
    return FastTTTAdapter(config.fast_ttt)


def _detached_norm(tensor: Tensor) -> float:
    return float(torch.linalg.vector_norm(tensor.detach().float()).cpu().item())


def _shares_storage(left: Tensor, right: Tensor) -> bool:
    if left.device.type == "meta" or right.device.type == "meta":
        return left is right
    return int(left.untyped_storage().data_ptr()) == int(right.untyped_storage().data_ptr())


def _assert_batched_state_storage_isolated(states: Sequence[FastWeightsState]) -> None:
    online_tensors = tuple(
        tensor
        for state in states
        for tensor in (state.w_t_1, state.w_t_2)
    )
    for left_index, left in enumerate(online_tensors):
        for right in online_tensors[left_index + 1 :]:
            if _shares_storage(left, right):
                raise ValueError("different Fast TTT batch items must not share W_t storage")
