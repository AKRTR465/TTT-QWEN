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
# A `_smooth_normalize`d non-degenerate row lands at norm ~1, and a zero row at
# exactly 0, so this only separates true degeneracy from mild shrinkage.
_DEGENERATE_ROW_NORM = 1.0e-3
# Centering removes ~99.4% of the slot norm, so restoring it needs a gain of
# ~150.  This ceiling only bites for a slot sitting numerically on the mean,
# where the direction is rounding noise and must not be amplified to full scale.
_MAX_CENTERING_GAIN = 1.0e3


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


PROBE_FIELDS: tuple[str, ...] = (
    "slot_pairwise",
    "value_raw_pairwise",
    "value_centered_pairwise",
    "key_centered_pairwise",
    "token_key_pairwise",
    "attention_entropy_ratio",
)


@dataclass(frozen=True, slots=True)
class SlotGeometryProbe:
    """Read-only slot-geometry diagnostics for one chunk of one video.

    Every field is a plain ``float`` bounded to ``[-1, 1]``: the reflection
    walkers that decide ``runtime_detached`` and the observation version
    signature recurse over dataclass fields, so carrying tensors here would
    silently flip those contracts.  Bounding the entropy ratio by ``log N``
    keeps one range predicate sufficient for the whole bundle.

    These answer which side of ``update = Delta^T diag(eta) K`` is collapsed.
    ``slot_pairwise`` is the decisive one: the slot states are the common
    ancestor of both keys (via the probe attention) and values (via ``W_v``),
    so if they are already collinear no memory-side change can help.
    """

    slot_pairwise: float = 0.0
    value_raw_pairwise: float = 0.0
    value_centered_pairwise: float = 0.0
    key_centered_pairwise: float = 0.0
    token_key_pairwise: float = 0.0
    attention_entropy_ratio: float = 0.0

    def __post_init__(self) -> None:
        for name in PROBE_FIELDS:
            value = getattr(self, name)
            if type(value) is not float:
                raise TypeError("slot geometry probes must be plain floats")
            if not math.isfinite(value) or not -1.0 <= value <= 1.0:
                raise ValueError("slot geometry probes must be finite in [-1, 1]")


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
    slot_geometry_probes: tuple[SlotGeometryProbe, ...] = ()

    def __post_init__(self) -> None:
        if self.slot_geometry_probes and len(self.slot_geometry_probes) != self.keys.shape[0]:
            raise ValueError("slot geometry probes must carry one entry per batch row")
        if any(
            not isinstance(probe, SlotGeometryProbe) for probe in self.slot_geometry_probes
        ):
            raise TypeError("slot geometry probes must be SlotGeometryProbe values")
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
                # The upper bound above says nothing about degeneracy, and a *valid*
                # slot whose row is zero is accepted everywhere downstream: the
                # delta is zero, the update is zero, the write still reports
                # did_write with write_norm 0.0, and the cosine audits return 0.0
                # rather than a NaN, so no existing check fires.  It is silent
                # capacity loss.  Fail closed instead -- callers that can produce a
                # degenerate row (subtracting a slot mean, for one) must handle it
                # explicitly rather than let it vanish.
                if bool(torch.any(norms[self.slot_mask] < _DEGENERATE_ROW_NORM)):
                    raise ValueError(
                        f"memory write {name} must not carry a zero-norm valid slot"
                    )


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
        ):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        with torch.no_grad():
            # The gate's output projection starts at zero, so every logit equals the
            # bias below and eta starts at exactly eta_gate_init on every slot, every
            # seed and any slot scale.  A Xavier weight here instead makes eta a
            # per-seed lottery: the gate reads raw slot states (norm ~9), silu is not
            # odd, and the slots are near-collinear, so W_out @ silu(W_h @ s) is one DC
            # offset *shared* by all slots rather than per-slot noise that averages
            # out -- drawn once per seed, near-constant across chunks, and growing
            # linearly with the slot norm.  That offset is what broke the
            # active_slots * eta_gate_init <= eta_chunk_budget guard: H200 drew
            # c >= +0.5 and renormalized 100% of chunks, pinning sum(eta) to exactly
            # the budget -- the delta rule's annihilation point -- while this box's
            # seed drew c ~ 0.1 and looked healthy.  W_out still trains normally
            # (d logit / d W_out = hidden != 0); only its init is pinned.
            nn.init.zeros_(self.memory_eta_gate_output.weight)
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
        base = F.linear(normalized, p_in_weight, p_in_bias)
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
        context_term = F.linear(
            normalized_context,
            p_context_weight,
            p_context_bias,
        ).unsqueeze(1)
        projected = base + context_term
        # The memory key space is the same affine map with the shared per-chunk
        # token mean of the *visual* component removed (see
        # _centered_over_valid_tokens for why that mean is the capacity ceiling).
        # This must be a separate tensor: ``projected`` also feeds the W0 core
        # below, and centering it in place would change the static function that
        # a zero memory is required to reproduce bitwise (the A2 preservation
        # invariant).  ``memory_keys`` feeds only the readout and the write path,
        # and with M = 0 the readout is exactly zero, so that invariant holds.
        # Both consumers share this one tensor, which keeps read keys, write
        # attention, and stored keys in a single space by construction.  The
        # exact ``context_term`` from above is re-added rather than recomputed:
        # accidentally dropping it would sever the only Bank->write-key gradient
        # channel, and the anisotropy-mean channel this centering removes is
        # replaced by that *intentional* per-chunk term, not by nothing.
        memory_keys = _centered_over_valid_tokens(base, mask) + context_term
        core_context = (
            torch.autocast(device_type=visual_embeddings.device.type, enabled=False)
            if visual_embeddings.device.type in {"cpu", "cuda"}
            else nullcontext()
        )
        readout_share_norms: tuple[float, ...]
        with core_context:
            projected = projected.float()
            memory_keys = memory_keys.float()
            hidden = F.linear(projected, w0_1_value.float(), None)
            core = F.linear(F.silu(hidden), w0_2_value.float(), None)
            if runtime_states is None:
                readout_share_norms = (0.0,) * visual_embeddings.shape[0]
            else:
                memory_stack = torch.stack([state.m for state in runtime_states])
                readout = alpha_value.float() * torch.bmm(
                    memory_keys,
                    memory_stack.transpose(1, 2),
                )
                readout_share_norms = _detached_readout_shares(readout, core, mask)
                core = core + readout
            predictions = core
        self._last_associative_intermediates = AssociativeTTTIntermediates(
            keys=memory_keys,
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
            if _slot_mean_diagnostic_enabled():
                # Under the slot-mean diagnostic the incoming bank is already exactly
                # uniform, so centering it again would yield the zero matrix and trip
                # the zero-norm guard in MemoryWriteBatch.  Skip the second centering
                # and let the write degenerate to the rank-1 payload a uniform bank
                # honestly implies: this diagnostic measures whether the *non-memory*
                # consumers use slot differentiation, and the write path only has to
                # stay alive, not stay meaningful.
                memory_slots = detached_slots
            else:
                memory_slots = _centered_over_valid_slots(detached_slots, slot_mask)
            probe = F.linear(
                memory_slots,
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
            raw_values = F.linear(
                memory_slots,
                self.memory_value_projection.weight.float(),
                self.memory_value_projection.bias.float(),
            )
            values = _smooth_normalize(raw_values)
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
            probes = _slot_geometry_probes(
                slots=detached_slots,
                raw_values=raw_values,
                pooled=pooled,
                token_keys=token_keys,
                attention=attention,
                slot_mask=slot_mask,
                token_mask=token_mask,
            )
        return MemoryWriteBatch(
            keys=keys,
            values=values,
            etas=etas,
            slot_mask=slot_mask,
            beta=beta,
            eta_renormalized=tuple(bool(value) for value in over_budget.detach().cpu().tolist()),
            slot_geometry_probes=probes,
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
        # Order is part of the optimizer-group contract: torch/DeepSpeed optimizer
        # state_dicts are keyed by position, so resume breaks if this is reordered.
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

    def collect_memory_write_parameters(self) -> tuple[nn.Parameter, ...]:
        """Return the parameters reachable *only* through the delta-rule write.

        These tensors appear nowhere outside :meth:`prepare_write`, so the Query
        deferred VJP is their sole gradient source.  Their joint gradient norm is
        therefore the one clean witness that the write mechanism is being trained
        rather than merely executed.  ``memory_alpha`` is deliberately excluded:
        it gates the read path and would keep looking healthy across a dead write
        path.  ``p_context`` is excluded for the same reason.
        """

        return (
            self.memory_key_probe.weight,
            self.memory_key_probe.bias,
            self.memory_value_projection.weight,
            self.memory_value_projection.bias,
            self.memory_eta_gate_hidden.weight,
            self.memory_eta_gate_hidden.bias,
            self.memory_eta_gate_output.weight,
            self.memory_eta_gate_output.bias,
            self.memory_beta_raw,
        )

    def collect_memory_read_parameters(self) -> tuple[nn.Parameter, ...]:
        """Return the read-side gate, whose gradient arrives via ``alpha (K Mᵀ)``."""

        return (self.memory_alpha,)

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


def _slot_mean_diagnostic_enabled() -> bool:
    """Read the opt-in slot-mean diagnostic gate, which lives with the slot encoder.

    Imported lazily so the two modules stay independent at import time; by the time a
    write is prepared both are loaded, so this is a dict lookup.
    """

    from ttt_svcbench_qwen.state_encoder import slot_mean_diagnostic_enabled

    return slot_mean_diagnostic_enabled()


def _centered_over_valid_slots(slots: Tensor, slot_mask: Tensor) -> Tensor:
    """Remove the component every slot of a chunk shares, per batch row.

    The slot states are the common ancestor of both halves of the write: keys
    come from ``W_k(slots)`` through the probe attention and values from
    ``W_v(slots)``.  Measured on 4xH200, ``slot_pairwise = 0.999223`` and
    ``value_raw_pairwise = 0.999205`` agree to five decimals -- ``W_v`` is very
    nearly angle-preserving, so the value geometry simply *is* the slot
    geometry.  On the key side the same collinearity makes all 32 probe vectors
    near-identical, so they rank tokens identically and the (already almost
    one-hot: entropy ratio 0.007-0.033) attention selects the *same* token for
    every slot, which is why the pooled keys land at 0.99968.

    Since ``rank(update) <= min(rank Delta, rank K)``, repairing only one side
    leaves the write rank-1 anyway.  Subtracting the cross-slot mean here fixes
    both at once, and it is the one intervention confined to the memory path:
    ``prepare_write`` consumes ``slots.detach()``, so the state heads, the Bank
    and the Reader are untouched, and ``forward`` never calls this at all so the
    ``M=0`` bitwise invariant cannot regress.

    Deliberately *not* centering the pooled keys instead: that would annihilate
    the per-chunk ``p_context`` broadcast, which is simultaneously the only
    source of cross-chunk key variation and the only path by which Bank
    conditioning reaches write-key geometry.  Centering upstream changes *which
    token each slot selects* while leaving that broadcast intact.

    Rows with fewer than two valid slots are returned unchanged: their mean is
    the slot itself, so centering would emit an exactly-zero payload row, which
    the write batch now rejects outright.

    Each slot's original norm is restored after centering, and that part is not
    cosmetic.  The shared component is ~99.4% of the slot norm, so centering
    alone shrinks the probe input ~150x, shrinks ``scores`` by the same factor,
    and flattens the softmax: measured, the attention entropy ratio jumped from
    0.033 to 0.981, i.e. every slot went back to pooling the same token *mean*
    instead of the same token.  That is the mirror image of the original failure
    and it caps the key gain at ~88x.  Rescaling to the original norm keeps the
    scores at their selective magnitude while the directions stay diverse, which
    is what lets different slots actually select different tokens.  The gain is
    capped so a slot sitting numerically on the mean cannot have its rounding
    noise amplified to full scale.
    """

    mask = slot_mask.unsqueeze(-1).to(dtype=slots.dtype)
    counts = mask.sum(dim=1, keepdim=True)
    mean = (slots * mask).sum(dim=1, keepdim=True) / counts.clamp_min(1.0)
    centered = slots - mean
    gain = (
        slots.norm(dim=-1, keepdim=True)
        / centered.norm(dim=-1, keepdim=True).clamp_min(_ETA_SCALE_TINY)
    ).clamp_max(_MAX_CENTERING_GAIN)
    centerable = counts >= 2.0
    return torch.where(centerable, centered * gain, slots)


def _centered_over_valid_tokens(tokens: Tensor, valid_mask: Tensor) -> Tensor:
    """Remove the component every valid token of a chunk shares, per batch row.

    The slot-side centering above raised realized write capacity 1.001 -> 4.213
    of 32 and stopped there because the memory keys are convex combinations of
    the *token* keys, whose own shared mean sets a hard ceiling: measured
    pairwise token cosine 0.46-0.50 caps the key participation ratio at ~4.2,
    and the pooled keys sit exactly on that ceiling.  The
    ``key_centered_pairwise`` probe puts the same keys at -0.027 once their
    shared component is removed -- i.e. the entire remaining capacity deficit
    *is* this mean.  Removing it lifts the ceiling to the value side's ~32.

    Only the visual component is centered.  The caller re-adds the per-chunk
    ``p_context`` broadcast afterwards, so the one cross-chunk key-variation
    source and the one Bank->write-key gradient channel both survive
    (``tests/test_associative_ttt.py`` guards the channel).  Centering the
    *summed* keys instead would annihilate that broadcast exactly, because it
    is constant across tokens within a chunk.

    Unlike the slot helper there is no norm restoration: the probe scores are
    ``probe @ keys^T``, so subtracting a per-chunk constant from every token
    shifts each slot's score row uniformly and the softmax over tokens is
    invariant -- token *selection* provably does not change, only the stored
    vectors do.  The per-token deviations that carry the selection signal are
    untouched, so there is no entropy collapse to compensate for.

    Rows with fewer than two valid tokens pass through unchanged: their mean is
    the token itself, and a centered single-token chunk would emit an exactly
    zero key that the write batch rejects as degenerate.
    """

    mask = valid_mask.unsqueeze(-1).to(dtype=tokens.dtype)
    counts = mask.sum(dim=1, keepdim=True)
    mean = (tokens * mask).sum(dim=1, keepdim=True) / counts.clamp_min(1.0)
    centerable = counts >= 2.0
    return torch.where(centerable, tokens - mean, tokens)


def _offdiag_cosine_mean(selected: Tensor) -> float:
    """Mean off-diagonal cosine over already-selected rows, in O(n*d).

    For unit rows ``sum_{i,j} u_i . u_j == ||sum_i u_i||^2``, so the
    off-diagonal mean is ``(||sum u||^2 - n) / (n(n-1))`` with no n-by-n Gram.
    That is what makes the token-level probe affordable at n ~ 10^3.

    Degenerate (zero-norm) rows are dropped *before* the identity is applied,
    because it is only valid when every row is unit.  Keeping them would make a
    fully collapsed set read ``-1/(n-1)`` -- numerically identical to the
    healthy floor of a centered equal-norm set, so the one measurement that has
    to distinguish "residual is fine" from "residual is gone" would report the
    same number for both.

    Fail policy deliberately mirrors
    ``memory_write._pairwise_offdiag_cosine_mean``: fewer than two usable rows
    and any non-finite result report ``0.0`` rather than raising.  These are
    diagnostics and must never take down a training step.
    """

    usable = selected[selected.norm(dim=-1) > _ETA_SCALE_TINY]
    count = int(usable.shape[0])
    if count < 2:
        return 0.0
    normalized = usable / usable.norm(dim=-1, keepdim=True)
    total = float(normalized.sum(dim=0).square().sum().item())
    value = (total - float(count)) / float(count * (count - 1))
    if not math.isfinite(value):
        return 0.0
    return max(-1.0, min(1.0, value))


def _centered_offdiag_cosine_mean(selected: Tensor) -> float:
    """Off-diagonal cosine after removing the shared component.

    A convex pool passes any row-invariant component through at coefficient
    exactly one, so the raw cosine is dominated by that shared direction.
    Subtracting the mean is the only way to see whether the residual carries
    anything.  For an equal-norm centered set the algebraic floor is
    ``-1/(n-1)``, so a reading near that floor means the residual is healthy
    and a reading still near ``+1`` means the residual is itself collapsed.
    """

    if int(selected.shape[0]) < 2:
        return 0.0
    return _offdiag_cosine_mean(selected - selected.mean(dim=0, keepdim=True))


def _attention_entropy_ratio(attention: Tensor, valid_tokens: int) -> float:
    """Mean slot-pool entropy normalized by ``log(valid tokens)``, in [0, 1].

    ``1.0`` means every slot pools uniformly over the same tokens, so all slots
    receive the same mean and the shared component dominates; ``0.0`` means
    one-hot.  Normalizing by ``log N`` is also what keeps this inside the one
    range predicate shared with the cosines.
    """

    if valid_tokens < 2:
        return 0.0
    safe = attention.clamp_min(_ETA_SCALE_TINY)
    entropy = -(attention * safe.log()).sum(dim=-1).mean()
    value = float(entropy.item()) / math.log(float(valid_tokens))
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


def _slot_geometry_probes(
    *,
    slots: Tensor,
    raw_values: Tensor,
    pooled: Tensor,
    token_keys: Tensor,
    attention: Tensor,
    slot_mask: Tensor,
    token_mask: Tensor,
) -> tuple[SlotGeometryProbe, ...]:
    """Measure where the write payload's geometry collapses, per batch row.

    Read-only.  ``prepare_write`` runs with grad *enabled*, so every tensor is
    detached here before use: attaching audit ops to the meta graph would
    retain memory across the truncation window and could open a spurious
    gradient path into the key probe or value projection, corrupting the
    ``collect_memory_write_parameters`` witness that proves the write path is
    trained.
    """

    if slots.device.type == "meta":
        return ()
    with torch.no_grad():
        detached_pooled = pooled.detach()
        detached_values = raw_values.detach()
        detached_slots = slots.detach()
        detached_tokens = token_keys.detach()
        detached_attention = attention.detach()
        probes: list[SlotGeometryProbe] = []
        for row in range(int(slots.shape[0])):
            rows_mask = slot_mask[row]
            tokens_mask = token_mask[row]
            row_slots = detached_slots[row][rows_mask]
            row_values = detached_values[row][rows_mask]
            row_pooled = detached_pooled[row][rows_mask]
            row_tokens = detached_tokens[row][tokens_mask]
            probes.append(
                SlotGeometryProbe(
                    slot_pairwise=_offdiag_cosine_mean(row_slots),
                    value_raw_pairwise=_offdiag_cosine_mean(row_values),
                    value_centered_pairwise=_centered_offdiag_cosine_mean(row_values),
                    key_centered_pairwise=_centered_offdiag_cosine_mean(row_pooled),
                    token_key_pairwise=_offdiag_cosine_mean(row_tokens),
                    attention_entropy_ratio=_attention_entropy_ratio(
                        detached_attention[row][rows_mask][:, tokens_mask],
                        int(row_tokens.shape[0]),
                    ),
                )
            )
    return tuple(probes)


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
