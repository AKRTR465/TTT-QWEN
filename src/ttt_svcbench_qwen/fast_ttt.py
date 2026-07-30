"""Implement the Fast TTT Adapter and its explicit per-video memory boundary.

Inputs: Main Merger embeddings, an optional padding mask, and explicit per-video
zero-initialized memory state.
Outputs: shape-preserving adapted embeddings plus detached numerical audit metadata,
and one per-chunk slot-derived write payload for the delta-rule memory.
Forbidden: memory writes, hidden memory registration, State Bank mutation, query
routing, gates over hard state, or online gradients into W0/RMSNorm/P_in/P_out.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ttt_svcbench_qwen.associative_ttt import (
    ASSOCIATIVE_CONTRACT_VERSION,
    BANK_EMBEDDING_DIM,
    AssociativeTTTIntermediates,
    FastAssociativeContext,
)
from ttt_svcbench_qwen.config import FastMemoryConfig, FastTTTConfig, ProjectConfig

MEMORY_DIM = 768
_NORMALIZE_EPSILON = 1.0e-6
_ETA_SCALE_TINY = 1.0e-12


class SlotStateView(Protocol):
    """Structural view of the spatial slot state consumed by one write payload."""

    @property
    def slots(self) -> Tensor: ...

    @property
    def slot_valid_mask(self) -> Tensor: ...

    @property
    def slot_confidence(self) -> Tensor: ...


@dataclass(frozen=True, slots=True)
class FastMemoryState:
    """One video's zero-initialized memory matrix; never register this on an nn.Module."""

    m: Tensor
    write_version: int
    write_count: int
    skip_count: int
    differentiable: bool = False

    def __post_init__(self) -> None:
        if self.m.shape != (MEMORY_DIM, MEMORY_DIM) or self.m.dtype != torch.float32:
            raise ValueError("memory master must be float32 [768, 768]")
        if self.m.device.type != "meta" and not bool(torch.isfinite(self.m).all()):
            raise ValueError("memory master must be finite")
        if not self.m.requires_grad:
            raise ValueError("memory master must require gradients")
        if type(self.differentiable) is not bool:
            raise TypeError("differentiable must be a bool")
        if not self.differentiable and not self.m.is_leaf:
            raise ValueError("non-differentiable runtime memory must be a leaf tensor")
        counters = (self.write_version, self.write_count, self.skip_count)
        if any(type(counter) is not int for counter in counters):
            raise TypeError("memory runtime counters must be exact integers")
        if min(counters) < 0:
            raise ValueError("memory runtime counters must be non-negative")
        if self.write_version != self.write_count:
            raise ValueError("write_version must equal the number of accepted writes")

    @property
    def fast_parameters(self) -> tuple[Tensor, ...]:
        return (self.m,)


@dataclass(frozen=True, slots=True)
class MemoryWriteBatch:
    """One chunk's adapter-derived write payload for a batch of videos.

    Keys/values are unit (or zero-padded) FP32 rows; per-slot write gains are
    non-negative and renormalized so each row's sum stays within the unit chunk
    budget — the precondition of the delta-rule contraction bound.
    """

    keys: Tensor
    values: Tensor
    etas: Tensor
    slot_mask: Tensor
    beta: Tensor
    eta_renormalized: tuple[bool, ...]

    def __post_init__(self) -> None:
        shape = self.keys.shape
        if len(shape) != 3 or shape[0] <= 0 or shape[1] <= 0 or shape[2] != MEMORY_DIM:
            raise ValueError("memory write keys must be non-empty [B, S, 768]")
        if self.values.shape != shape:
            raise ValueError("memory write values must align with keys")
        if self.etas.shape != shape[:2]:
            raise ValueError("memory write etas must be [B, S]")
        if self.slot_mask.shape != shape[:2] or self.slot_mask.dtype != torch.bool:
            raise ValueError("memory write slot_mask must be bool [B, S]")
        floating = (self.keys, self.values, self.etas, self.beta)
        if any(value.dtype != torch.float32 for value in floating):
            raise ValueError("memory write payload must be FP32")
        if any(value.device != self.keys.device for value in (*floating[1:], self.slot_mask)):
            raise ValueError("memory write payload must share one device")
        if self.beta.ndim != 0:
            raise ValueError("memory write beta must be one scalar")
        if len(self.eta_renormalized) != shape[0] or any(
            type(flag) is not bool for flag in self.eta_renormalized
        ):
            raise ValueError("memory write eta_renormalized must be bool per batch row")
        if self.keys.device.type == "meta":
            return
        if any(not bool(torch.isfinite(value).all()) for value in floating):
            raise ValueError("memory write payload must be finite")
        beta_value = float(self.beta.detach().item())
        if not 0.0 < beta_value < 1.0:
            raise ValueError("memory write beta must lie strictly inside (0, 1)")
        with torch.no_grad():
            etas = self.etas.detach()
            if bool(torch.any(etas < 0.0)):
                raise ValueError("memory write etas must be non-negative")
            if bool(torch.any(etas[~self.slot_mask] != 0.0)):
                raise ValueError("invalid slots must carry exactly zero eta")
            if bool(torch.any(etas.sum(dim=1) > 1.0 + 1.0e-4)):
                raise ValueError("memory write eta sums exceed the unit chunk budget")
            for name, rows in (("keys", self.keys), ("values", self.values)):
                norms = rows.detach().norm(dim=-1)
                if bool(torch.any(norms > 1.0 + 1.0e-3)):
                    raise ValueError(f"memory write {name} must not exceed unit row norm")
                if bool(torch.any(norms[~self.slot_mask] != 0.0)):
                    raise ValueError(f"memory write {name} must zero-pad invalid slots")


@dataclass(frozen=True, slots=True)
class FastParameterGroups:
    """Separate checkpointed static-core, checkpointed slow, and transient online tensors."""

    meta_fast: tuple[nn.Parameter, nn.Parameter]
    slow: tuple[nn.Parameter, ...]
    online_fast: tuple[Tensor, ...]


@dataclass(frozen=True, slots=True)
class FastQueryProxyAudit:
    """Detached evidence for one Query-local memory proxy."""

    write_version: int
    write_count: int
    skip_count: int
    max_abs_value_drift: float
    storage_isolated: bool
    graph_detached: bool
    leaf_fast_weights: bool

    def __post_init__(self) -> None:
        counters = (self.write_version, self.write_count, self.skip_count)
        if any(type(value) is not int or value < 0 for value in counters):
            raise ValueError("Fast Query proxy counters must be non-negative integers")
        if self.write_version != self.write_count:
            raise ValueError("Fast Query proxy version must equal accepted writes")
        if not math.isfinite(self.max_abs_value_drift) or self.max_abs_value_drift < 0.0:
            raise ValueError("Fast Query proxy drift must be finite and non-negative")
        if not self.storage_isolated or not self.graph_detached or not self.leaf_fast_weights:
            raise ValueError("Fast Query proxy must be detached, leaf, and storage-isolated")


@dataclass(frozen=True, slots=True)
class FastTTTForwardAudit:
    write_versions: tuple[int, ...]
    write_counts: tuple[int, ...]
    valid_token_counts: tuple[int, ...]
    used_runtime_state: bool
    used_associative_context: bool
    bank_record_counts: tuple[int, ...]
    memory_norms: tuple[float, ...]
    readout_share_norms: tuple[float, ...]
    input_norms: tuple[float, ...]
    residual_norms: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            type(self.used_runtime_state) is not bool
            or type(self.used_associative_context) is not bool
        ):
            raise TypeError("Fast TTT runtime/context audit flags must be bool")
        lengths = {
            len(self.write_versions),
            len(self.write_counts),
            len(self.valid_token_counts),
            len(self.bank_record_counts),
            len(self.memory_norms),
            len(self.readout_share_norms),
            len(self.input_norms),
            len(self.residual_norms),
        }
        if lengths != {len(self.write_versions)} or not self.write_versions:
            raise ValueError("Fast TTT audit fields must align to one non-empty batch")
        counters = (
            *self.write_versions,
            *self.write_counts,
            *self.valid_token_counts,
            *self.bank_record_counts,
        )
        if any(type(counter) is not int for counter in counters):
            raise TypeError("Fast TTT audit counters must be exact integers")
        if min(counters) < 0:
            raise ValueError("Fast TTT audit counters must be non-negative")
        if min(self.valid_token_counts) == 0:
            raise ValueError("Fast TTT audit requires at least one valid token per batch item")
        if self.write_versions != self.write_counts:
            raise ValueError("Fast TTT audit version must equal accepted write count")
        values = (
            *self.memory_norms,
            *self.readout_share_norms,
            *self.input_norms,
            *self.residual_norms,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("Fast TTT audit norms must be finite and non-negative")


class FastTTTAdapter(nn.Module):  # type: ignore[misc]
    """4096→768→768→4096 residual Adapter with an external per-video memory matrix."""

    def __init__(self, config: FastTTTConfig, memory: FastMemoryConfig) -> None:
        super().__init__()
        if config.fast_bias:
            raise ValueError("Fast TTT matrices must not use bias")
        if not config.slow_projection_bias:
            raise ValueError("P5 parameter budget requires bias on the slow projections")
        if config.fast_initialization != "xavier_uniform":
            raise ValueError("P5 requires xavier_uniform W0 initialization")
        if config.bottleneck_dim != MEMORY_DIM:
            raise ValueError("slot memory requires the 768 fast bottleneck")
        if config.fast_matrix_count != 1:
            raise ValueError("slot memory carries exactly one online matrix")
        if config.online_parameter_count != MEMORY_DIM * MEMORY_DIM:
            raise ValueError("fast_ttt.online_parameter_count must equal one memory matrix")
        if memory.write_rule != "parallel_delta_rule":
            raise ValueError("slot memory requires the parallel delta-rule write contract")
        if memory.memory_dtype != "float32" or not memory.zero_init_per_video:
            raise ValueError("slot memory must be zero-initialized FP32 per video")
        self.config = config
        self.memory_config = memory
        self.input_dim = config.input_dim
        self.bottleneck_dim = config.bottleneck_dim
        self.output_dim = config.output_dim
        self.residual_scale = config.residual_scale
        self.eta_max = float(memory.eta_max_per_slot)
        self.eta_budget = float(memory.eta_chunk_budget)
        self.beta_max = float(memory.forget_beta_max)
        self.rms_norm = nn.RMSNorm(config.input_dim, eps=config.rms_norm_eps)
        self.p_in = nn.Linear(
            config.input_dim,
            config.bottleneck_dim,
            bias=config.slow_projection_bias,
        )
        self.p_context = nn.Linear(
            BANK_EMBEDDING_DIM,
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
        self.memory_key_probe = nn.Linear(MEMORY_DIM, MEMORY_DIM, bias=True)
        self.memory_value_projection = nn.Linear(MEMORY_DIM, MEMORY_DIM, bias=True)
        self.memory_eta_gate_hidden = nn.Linear(MEMORY_DIM + 1, memory.eta_gate_hidden_dim)
        self.memory_eta_gate_output = nn.Linear(memory.eta_gate_hidden_dim, 1)
        for module in (
            self.memory_key_probe,
            self.memory_value_projection,
            self.memory_eta_gate_hidden,
            self.memory_eta_gate_output,
        ):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        with torch.no_grad():
            # sigma(bias) * eta_max == eta_gate_init at a zero-centered gate input.
            gate_ratio = memory.eta_gate_init / memory.eta_max_per_slot
            self.memory_eta_gate_output.bias.fill_(math.log(gate_ratio / (1.0 - gate_ratio)))
        self.memory_alpha = nn.Parameter(
            torch.full((MEMORY_DIM,), float(memory.read_gate_init))
        )
        beta_ratio = memory.forget_beta_init / memory.forget_beta_max
        self.memory_beta_raw = nn.Parameter(
            torch.tensor(math.log(beta_ratio / (1.0 - beta_ratio)))
        )
        self.register_buffer(
            "memory_contract_version",
            torch.tensor(ASSOCIATIVE_CONTRACT_VERSION, dtype=torch.int64),
            persistent=True,
        )
        self.reset_associative_projections()
        self._active_fast_states: tuple[FastMemoryState, ...] | None = None
        self._active_associative_context: FastAssociativeContext | None = None
        self._last_associative_intermediates: AssociativeTTTIntermediates | None = None
        self.last_audit: FastTTTForwardAudit | None = None

    def forward(
        self,
        visual_embeddings: Tensor,
        valid_mask: Tensor | None = None,
        metadata: object | None = None,
        *,
        fast_state: FastMemoryState | Sequence[FastMemoryState] | None = None,
    ) -> Tensor:
        """Run the static W0 core, adding the per-video memory readout when bound.

        A zero memory is bitwise-identical to the static-only forward: the A2
        initialization is exactly preserved at every episode start.
        """

        del metadata
        self.last_audit = None
        self._last_associative_intermediates = None
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
        detach_slow = False
        if runtime_states is not None:
            for state in runtime_states:
                self._validate_state_for_module(state)
            differentiable_modes = {state.differentiable for state in runtime_states}
            if len(differentiable_modes) != 1:
                raise ValueError("one Fast TTT batch cannot mix differentiable and online states")
            detach_slow = not runtime_states[0].differentiable
        write_versions = (
            (0,) * visual_embeddings.shape[0]
            if runtime_states is None
            else tuple(state.write_version for state in runtime_states)
        )
        write_counts = (
            (0,) * visual_embeddings.shape[0]
            if runtime_states is None
            else tuple(state.write_count for state in runtime_states)
        )

        rms_weight = self._online_value(self.rms_norm.weight, detach_slow)
        p_in_weight = self._online_value(self.p_in.weight, detach_slow)
        p_in_bias = self._online_value(self.p_in.bias, detach_slow)
        p_context_weight = self._online_value(self.p_context.weight, detach_slow)
        p_context_bias = self._online_value(self.p_context.bias, detach_slow)
        p_out_weight = self._online_value(self.p_out.weight, detach_slow)
        p_out_bias = self._online_value(self.p_out.bias, detach_slow)
        w0_1_value = self._online_value(self.w0_1, detach_slow)
        w0_2_value = self._online_value(self.w0_2, detach_slow)
        alpha_value = self._online_value(self.memory_alpha, detach_slow)
        assert rms_weight is not None
        assert w0_1_value is not None and w0_2_value is not None and alpha_value is not None
        normalized = F.rms_norm(
            visual_embeddings,
            (self.input_dim,),
            rms_weight,
            self.config.rms_norm_eps,
        )
        projected = F.linear(normalized, p_in_weight, p_in_bias)
        context = self._active_associative_context
        if context is None:
            combined_query = visual_embeddings.new_zeros(
                (visual_embeddings.shape[0], BANK_EMBEDDING_DIM)
            )
            bank_record_counts = torch.zeros(
                visual_embeddings.shape[0],
                dtype=torch.int64,
                device=visual_embeddings.device,
            )
            bank_versions = (0,) * visual_embeddings.shape[0]
        else:
            context = context.to(visual_embeddings)
            if context.combined_query.shape[0] != visual_embeddings.shape[0]:
                raise ValueError("associative context and visual batch must align")
            combined_query = context.combined_query
            bank_record_counts = context.bank_record_counts
            bank_versions = context.bank_versions
        normalized_context = F.layer_norm(
            combined_query,
            (BANK_EMBEDDING_DIM,),
            None,
            None,
        )
        projected = projected + F.linear(
            normalized_context,
            p_context_weight,
            p_context_bias,
        ).unsqueeze(1)
        core_context = (
            torch.autocast(device_type=visual_embeddings.device.type, enabled=False)
            if visual_embeddings.device.type in {"cpu", "cuda"}
            else nullcontext()
        )
        readout_share_norms: tuple[float, ...]
        with core_context:
            projected = projected.float()
            hidden = F.linear(projected, w0_1_value.float(), None)
            core = F.linear(F.silu(hidden), w0_2_value.float(), None)
            if runtime_states is None:
                readout_share_norms = (0.0,) * visual_embeddings.shape[0]
            else:
                memory_stack = torch.stack([state.m for state in runtime_states])
                readout = alpha_value.float() * torch.bmm(
                    projected,
                    memory_stack.transpose(1, 2),
                )
                readout_share_norms = _detached_readout_shares(readout, core, mask)
                core = core + readout
            predictions = core
        self._last_associative_intermediates = AssociativeTTTIntermediates(
            keys=projected,
            predictions=predictions,
            valid_mask=mask,
            bank_record_counts=bank_record_counts,
            bank_versions=bank_versions,
        )
        residual = F.linear(
            predictions.to(dtype=visual_embeddings.dtype),
            p_out_weight,
            p_out_bias,
        )
        residual = residual.masked_fill(~mask.unsqueeze(-1), 0.0)
        scaled_residual = self.residual_scale * residual
        output = visual_embeddings + scaled_residual
        if not bool(torch.isfinite(output).all()):
            raise ValueError("Fast TTT output must be finite")
        memory_norms = (
            (0.0,) * visual_embeddings.shape[0]
            if runtime_states is None
            else tuple(_detached_norm(state.m) for state in runtime_states)
        )
        self.last_audit = FastTTTForwardAudit(
            write_versions=write_versions,
            write_counts=write_counts,
            valid_token_counts=tuple(int(row.sum().item()) for row in mask),
            used_runtime_state=runtime_states is not None,
            used_associative_context=context is not None,
            bank_record_counts=tuple(int(value.item()) for value in bank_record_counts),
            memory_norms=memory_norms,
            readout_share_norms=readout_share_norms,
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

    def prepare_write(
        self,
        intermediates: AssociativeTTTIntermediates,
        spatial: SlotStateView,
    ) -> MemoryWriteBatch:
        """Derive one chunk's (key, value, eta) slot payload from committed soft state.

        Keys are probe-attention pools over this chunk's live token keys, so the
        outer loop trains memory key geometry through P_in/P_C.  Slot states and
        confidences enter only through stop-gradient: the write is archival and
        must not pull encoder representations toward the memory.
        """

        if not isinstance(intermediates, AssociativeTTTIntermediates):
            raise TypeError("prepare_write requires AssociativeTTTIntermediates")
        slots = spatial.slots
        slot_mask = spatial.slot_valid_mask
        confidence = spatial.slot_confidence
        if (
            slots.ndim != 3
            or slots.shape[0] != intermediates.keys.shape[0]
            or slots.shape[2] != MEMORY_DIM
            or not torch.is_floating_point(slots)
        ):
            raise ValueError("prepare_write requires slot states [B, S, 768]")
        if slot_mask.shape != slots.shape[:2] or slot_mask.dtype != torch.bool:
            raise ValueError("prepare_write requires bool slot_valid_mask [B, S]")
        if confidence.shape != slots.shape[:2] or not torch.is_floating_point(confidence):
            raise ValueError("prepare_write requires floating slot_confidence [B, S]")
        if slots.device != intermediates.keys.device:
            raise ValueError("prepare_write slot state and token keys must share one device")
        core_context = (
            torch.autocast(device_type=slots.device.type, enabled=False)
            if slots.device.type in {"cpu", "cuda"}
            else nullcontext()
        )
        with core_context:
            token_keys = intermediates.keys.float()
            token_mask = intermediates.valid_mask
            detached_slots = slots.detach().float()
            detached_confidence = confidence.detach().float()
            probe = F.linear(
                detached_slots,
                self.memory_key_probe.weight.float(),
                self.memory_key_probe.bias.float(),
            )
            scores = torch.bmm(probe, token_keys.transpose(1, 2)) / math.sqrt(MEMORY_DIM)
            scores = scores.masked_fill(
                ~token_mask.unsqueeze(1),
                torch.finfo(scores.dtype).min,
            )
            attention = torch.softmax(scores, dim=-1)
            pooled = torch.bmm(attention, token_keys)
            keys = _smooth_normalize(pooled)
            values = _smooth_normalize(
                F.linear(
                    detached_slots,
                    self.memory_value_projection.weight.float(),
                    self.memory_value_projection.bias.float(),
                )
            )
            gate_inputs = torch.cat(
                (detached_slots, detached_confidence.unsqueeze(-1)),
                dim=-1,
            )
            gate_hidden = F.silu(
                F.linear(
                    gate_inputs,
                    self.memory_eta_gate_hidden.weight.float(),
                    self.memory_eta_gate_hidden.bias.float(),
                )
            )
            gate_logits = F.linear(
                gate_hidden,
                self.memory_eta_gate_output.weight.float(),
                self.memory_eta_gate_output.bias.float(),
            ).squeeze(-1)
            slot_scale = slot_mask.to(dtype=torch.float32)
            etas = self.eta_max * torch.sigmoid(gate_logits) * slot_scale
            eta_sums = etas.sum(dim=1)
            over_budget = eta_sums > self.eta_budget
            renormalizer = torch.where(
                over_budget,
                self.eta_budget / eta_sums.clamp_min(_ETA_SCALE_TINY),
                torch.ones_like(eta_sums),
            )
            etas = etas * renormalizer.unsqueeze(-1)
            keys = keys * slot_scale.unsqueeze(-1)
            values = values * slot_scale.unsqueeze(-1)
            beta = self.beta_max * torch.sigmoid(self.memory_beta_raw.float())
        return MemoryWriteBatch(
            keys=keys,
            values=values,
            etas=etas,
            slot_mask=slot_mask,
            beta=beta,
            eta_renormalized=tuple(bool(value) for value in over_budget.detach().cpu().tolist()),
        )

    def initialize_fast_state(self, *, differentiable: bool = False) -> FastMemoryState:
        """Create one zero, storage-independent per-video memory matrix.

        Zero initialization is structural: the memory cannot carry static
        capacity across videos, so anything it contributes at query time was
        necessarily written during this video.
        """

        memory = torch.zeros(
            (MEMORY_DIM, MEMORY_DIM),
            dtype=torch.float32,
            device=self.w0_1.device,
            requires_grad=True,
        )
        return FastMemoryState(
            m=memory,
            write_version=0,
            write_count=0,
            skip_count=0,
            differentiable=differentiable,
        )

    def reset_fast_state(
        self,
        state: FastMemoryState | None = None,
        *,
        differentiable: bool | None = None,
    ) -> FastMemoryState:
        """Reset counters and zero the memory for a fresh video episode."""

        if differentiable is not None and type(differentiable) is not bool:
            raise TypeError("differentiable reset mode must be a bool")
        mode = (
            state.differentiable if state is not None and differentiable is None else differentiable
        )
        return self.initialize_fast_state(differentiable=bool(mode))

    @contextmanager
    def use_fast_state(
        self,
        state: FastMemoryState | Sequence[FastMemoryState],
    ) -> Iterator[FastTTTAdapter]:
        """Temporarily bind memory state for the unchanged P3 Qwen adapter call signature."""

        if self._active_fast_states is not None:
            raise RuntimeError("Fast TTT runtime state binding is not re-entrant")
        states = (state,) if isinstance(state, FastMemoryState) else tuple(state)
        if not states:
            raise ValueError("Fast TTT runtime state binding cannot be empty")
        if not all(isinstance(item, FastMemoryState) for item in states):
            raise TypeError("Fast TTT bindings must contain only FastMemoryState values")
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

    @contextmanager
    def use_associative_context(
        self,
        context: FastAssociativeContext,
    ) -> Iterator[FastTTTAdapter]:
        """Bind one immutable pre-write Bank context to the next visual call(s)."""

        if not isinstance(context, FastAssociativeContext):
            raise TypeError("associative context binding requires FastAssociativeContext")
        if self._active_associative_context is not None:
            raise RuntimeError("Fast associative context binding is not re-entrant")
        self._active_associative_context = context
        try:
            yield self
        except BaseException:
            self._last_associative_intermediates = None
            raise
        finally:
            self._active_associative_context = None

    def consume_associative_intermediates(self) -> AssociativeTTTIntermediates:
        """Take and clear the ephemeral tensors captured by the latest visual call."""

        value = self._last_associative_intermediates
        self._last_associative_intermediates = None
        if value is None:
            raise RuntimeError("no associative intermediates are available to consume")
        return value

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def reset_associative_projections(self) -> None:
        """Initialize the Bank-conditioned context path without changing W0."""

        self.p_context.weight.zero_()
        if self.p_context.bias is not None:
            self.p_context.bias.zero_()

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

    def collect_associative_parameters(self) -> tuple[nn.Parameter, ...]:
        context_bias = self.p_context.bias
        if context_bias is None:
            raise RuntimeError("associative projection biases disappeared")
        return (
            self.p_context.weight,
            context_bias,
            self.memory_key_probe.weight,
            self.memory_key_probe.bias,
            self.memory_value_projection.weight,
            self.memory_value_projection.bias,
            self.memory_eta_gate_hidden.weight,
            self.memory_eta_gate_hidden.bias,
            self.memory_eta_gate_output.weight,
            self.memory_eta_gate_output.bias,
            self.memory_alpha,
            self.memory_beta_raw,
        )

    def parameter_groups(self, state: FastMemoryState) -> FastParameterGroups:
        return FastParameterGroups(
            meta_fast=self.collect_meta_fast_parameters(),
            slow=self.collect_slow_parameters(),
            online_fast=collect_fast_parameters(state),
        )

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

    def _validate_state_for_module(self, state: FastMemoryState) -> None:
        if state.m.device != self.w0_1.device:
            raise ValueError("Fast TTT runtime memory must live on the module device")

    def _normalize_runtime_states(
        self,
        state: FastMemoryState | Sequence[FastMemoryState] | None,
        batch_size: int,
    ) -> tuple[FastMemoryState, ...] | None:
        if state is None:
            return None
        states = (state,) if isinstance(state, FastMemoryState) else tuple(state)
        if not all(isinstance(item, FastMemoryState) for item in states):
            raise TypeError("Fast TTT state sequences must contain only FastMemoryState values")
        if len(states) != batch_size:
            raise ValueError("Fast TTT requires exactly one runtime state per batch item")
        _assert_batched_state_storage_isolated(states)
        return states

    @staticmethod
    def _online_value(value: Tensor | None, detach: bool) -> Tensor | None:
        if value is None:
            return None
        return value.detach() if detach else value


def collect_fast_parameters(state: FastMemoryState) -> tuple[Tensor, ...]:
    """Return only the online memory matrix, in stable formula order."""

    return state.fast_parameters


def make_query_proxy_fast_state(
    state: FastMemoryState,
) -> tuple[FastMemoryState, FastQueryProxyAudit]:
    """Create an isolated leaf proxy with the exact numeric value of differentiable ``M``.

    Query-local backward may consume and release this proxy graph immediately.  A later
    deferred VJP explicitly links the captured proxy gradient back to the authoritative
    differentiable memory state, so this helper must not retain any edge to the Support
    graph.
    """

    if not isinstance(state, FastMemoryState):
        raise TypeError("make_query_proxy_fast_state requires one FastMemoryState")
    if not state.differentiable:
        raise ValueError("Query proxy requires a differentiable meta-training memory state")

    old_value = state.m.detach()
    proxy = old_value.clone().requires_grad_(True)
    proxy_state = FastMemoryState(
        m=proxy,
        write_version=state.write_version,
        write_count=state.write_count,
        skip_count=state.skip_count,
        differentiable=True,
    )
    isolated = not _shares_storage(proxy, state.m)
    if state.m.device.type == "meta":
        drift = 0.0
    else:
        drift = float((proxy.detach() - old_value).abs().max().float().cpu().item())
    audit = FastQueryProxyAudit(
        write_version=state.write_version,
        write_count=state.write_count,
        skip_count=state.skip_count,
        max_abs_value_drift=drift,
        storage_isolated=isolated,
        graph_detached=proxy.grad_fn is None,
        leaf_fast_weights=proxy.is_leaf,
    )
    return proxy_state, audit


def deferred_fast_vjp_loss(
    authoritative_states: Sequence[FastMemoryState],
    proxy_gradients: Sequence[Tensor],
) -> Tensor:
    """Return a numerically-zero scalar whose gradient injects a deferred memory VJP."""

    states = tuple(authoritative_states)
    gradients = tuple(proxy_gradients)
    authoritative = tuple(value for state in states for value in state.fast_parameters)
    if not states or len(gradients) != len(authoritative):
        raise ValueError("deferred fast VJP states and gradients must be non-empty and aligned")
    if any(not state.differentiable for state in states):
        raise ValueError("deferred fast VJP requires differentiable authoritative states")
    if any(not value.requires_grad for value in authoritative):
        raise ValueError("deferred fast VJP authoritative matrices must require gradients")
    terms: list[Tensor] = []
    for value, gradient in zip(authoritative, gradients, strict=True):
        if (
            gradient.shape != value.shape
            or gradient.device != value.device
            or not torch.is_floating_point(gradient)
        ):
            raise ValueError("deferred fast VJP gradient shape/device/dtype contract drifted")
        if gradient.requires_grad or gradient.grad_fn is not None:
            raise ValueError("deferred fast VJP gradients must be detached cotangents")
        if gradient.device.type != "meta" and not bool(torch.isfinite(gradient).all()):
            raise ValueError("deferred fast VJP gradients must be finite")
        accumulation_dtype = torch.promote_types(value.dtype, gradient.dtype)
        terms.append(
            (
                value.to(dtype=accumulation_dtype)
                * gradient.to(dtype=accumulation_dtype)
            ).sum()
        )
    link = torch.stack(terms).sum()
    if link.ndim != 0 or not link.requires_grad:
        raise ValueError("deferred fast VJP link must remain one differentiable scalar")
    # Preserve only d<link>/dM.  The arbitrary dot-product value is not part of L_total.
    return link - link.detach()


def build_fast_ttt_adapter(config: ProjectConfig | None = None) -> FastTTTAdapter:
    if config is None:
        raise ValueError("build_fast_ttt_adapter requires a validated ProjectConfig")
    return FastTTTAdapter(config.fast_ttt, config.fast_memory)


def _smooth_normalize(value: Tensor) -> Tensor:
    """Normalize with a finite first/second derivative at the zero vector."""

    inverse_norm = torch.rsqrt(
        value.square().sum(dim=-1, keepdim=True) + _NORMALIZE_EPSILON**2
    )
    return value * inverse_norm


def _detached_readout_shares(
    readout: Tensor,
    static_core: Tensor,
    valid_mask: Tensor,
) -> tuple[float, ...]:
    """Measure per-row ||alpha (K M^T)|| / ||f_W0(K)|| without retaining an audit graph."""

    if readout.device.type == "meta":
        return (0.0,) * int(readout.shape[0])
    values: list[float] = []
    with torch.no_grad():
        for row in range(readout.shape[0]):
            mask = valid_mask[row]
            readout_norm = torch.linalg.vector_norm(readout[row][mask].detach().float())
            core_norm = torch.linalg.vector_norm(static_core[row][mask].detach().float())
            values.append(
                float((readout_norm / core_norm.clamp_min(1.0e-12)).cpu().item())
            )
    return tuple(values)


def _detached_norm(tensor: Tensor) -> float:
    if tensor.device.type == "meta":
        return 0.0
    return float(torch.linalg.vector_norm(tensor.detach().float()).cpu().item())


def _shares_storage(left: Tensor, right: Tensor) -> bool:
    if left.device.type == "meta" or right.device.type == "meta":
        return left is right
    if left.device != right.device or left.numel() == 0 or right.numel() == 0:
        return False
    if int(left.untyped_storage().data_ptr()) != int(right.untyped_storage().data_ptr()):
        return False

    # ZeRO may repoint multiple Parameters at disjoint views of one flattened
    # allocation.  A shared base pointer alone is therefore not evidence that
    # the live tensor regions alias.  Compare the byte spans addressed by each
    # strided view; overlapping spans remain a fail-closed result.
    left_start, left_end = _storage_byte_span(left)
    right_start, right_end = _storage_byte_span(right)
    return left_start < right_end and right_start < left_end


def _storage_byte_span(tensor: Tensor) -> tuple[int, int]:
    minimum = tensor.storage_offset()
    maximum = minimum
    for size, stride in zip(tensor.shape, tensor.stride(), strict=True):
        delta = (size - 1) * stride
        minimum += min(0, delta)
        maximum += max(0, delta)
    element_size = tensor.element_size()
    return minimum * element_size, (maximum + 1) * element_size


def _assert_batched_state_storage_isolated(states: Sequence[FastMemoryState]) -> None:
    online_tensors = tuple(state.m for state in states)
    for left_index, left in enumerate(online_tensors):
        for right in online_tensors[left_index + 1 :]:
            if _shares_storage(left, right):
                raise ValueError("different Fast TTT batch items must not share memory storage")
