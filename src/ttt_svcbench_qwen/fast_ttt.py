"""Fast TTT Adapter, its Bank-conditioned key context, and the delta-rule memory write.

Inputs: Main Merger embeddings, an optional padding mask, and explicit per-video
zero-initialized memory state.
Outputs: shape-preserving adapted embeddings, one per-chunk slot-derived write
payload, and the closed-form parallel delta-rule write that consumes it.
Forbidden: hidden memory registration, State Bank mutation, query routing, gates
over hard state, or online gradients into W0/RMSNorm/P_in/P_out.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ttt_svcbench_qwen.config import FastMemoryConfig, FastTTTConfig, ProjectConfig
from ttt_svcbench_qwen.state_bank import StateBankView

MEMORY_DIM = 768
_NORMALIZE_EPSILON = 1.0e-6
_ETA_SCALE_TINY = 1.0e-12
# Centering removes ~99.4% of the slot norm, so restoring it needs a gain of
# ~150.  This ceiling only bites for a slot sitting numerically on the mean,
# where the direction is rounding noise and must not be amplified to full scale.
_MAX_CENTERING_GAIN = 1.0e3

ASSOCIATIVE_CONTRACT = "bank_conditioned_slot_memory_v3"
# Revision 4: memory keys are the token-centered visual projection plus the
# p_context broadcast (_centered_over_valid_tokens).  The contract *family*
# string above is a config literal pinned across four sites and the mechanism it
# names is unchanged, so only this revision counter moves; it is what the
# warmup-bundle provenance pins, so bundles produced under revision 3's
# uncentered key space refuse to load instead of silently mixing key semantics.
ASSOCIATIVE_CONTRACT_VERSION = 4
BANK_EMBEDDING_DIM = 512
ASSOCIATIVE_DIM = 768


class SlotStateView(Protocol):
    """Structural view of the spatial slot state consumed by one write payload."""

    @property
    def slots(self) -> Tensor: ...

    @property
    def slot_valid_mask(self) -> Tensor: ...

    @property
    def slot_confidence(self) -> Tensor: ...


class StateWriteSourceView(Protocol):
    """Minimal soft-write surface still produced by the heads for the Bank path."""

    @property
    def o1_present_mask(self) -> Tensor: ...

    @property
    def o2_present_mask(self) -> Tensor: ...

    @property
    def e1_present_mask(self) -> Tensor: ...

    @property
    def e2_present_mask(self) -> Tensor: ...

    @property
    def o1_sources(self) -> Tensor: ...

    @property
    def o2_sources(self) -> Tensor: ...

    @property
    def e1_sources(self) -> Tensor: ...

    @property
    def e2_sources(self) -> Tensor: ...


@dataclass(frozen=True, slots=True)
class FastMemoryState:
    """One video's zero-initialized memory matrix; never register this on an nn.Module.

    ``differentiable`` is mechanism, not metadata: it selects enable_grad vs
    no_grad around the delta rule and whether ``_next_memory`` clones or re-leafs.
    The three counters are the memory generation index that inference and the K=8
    meta boundary thread forward.
    """

    m: Tensor
    write_version: int
    write_count: int
    skip_count: int
    differentiable: bool = False

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


@dataclass(frozen=True, slots=True)
class FastTTTForwardAudit:
    """Detached per-forward evidence; ``used_runtime_state`` gates the M4 read path."""

    write_versions: tuple[int, ...]
    write_counts: tuple[int, ...]
    valid_token_counts: tuple[int, ...]
    used_runtime_state: bool
    bank_record_counts: tuple[int, ...]
    readout_share_norms: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class FastAssociativeContext:
    """Immutable, pre-write Bank context consumed by one visual forward."""

    combined_query: Tensor
    bank_record_counts: Tensor
    bank_versions: tuple[int, ...]

    def to(self, reference: Tensor) -> FastAssociativeContext:
        """Move only the immutable tensor view to the visual reference."""

        return FastAssociativeContext(
            combined_query=self.combined_query.to(
                device=reference.device,
                dtype=reference.dtype,
            ),
            bank_record_counts=self.bank_record_counts.to(device=reference.device),
            bank_versions=self.bank_versions,
        )


@dataclass(frozen=True, slots=True)
class AssociativeTTTIntermediates:
    """Ephemeral key/prediction tensors captured by one adapter call."""

    keys: Tensor
    predictions: Tensor
    valid_mask: Tensor
    bank_record_counts: Tensor
    bank_versions: tuple[int, ...]


class MemoryWriteSkipReason(StrEnum):
    NO_VALID_SLOT = "no_valid_slot"
    NONFINITE_KEY_VALUE = "nonfinite_key_value"


class GradientMode(StrEnum):
    """Exact autograd contract used by one delta-rule write."""

    ONLINE_LEAF = "online_leaf"
    META_LINEAR_RECURRENCE = "meta_linear_recurrence"


@dataclass(frozen=True, slots=True)
class MemoryWriteResult:
    """One chunk's write outcome.  ``did_write`` drives the recurrence itself."""

    fast_state: FastMemoryState
    did_write: bool
    slots_written: int
    eta_sum: float
    eta_renormalized: bool
    write_norm: float
    memory_norm: float
    skip_reason: MemoryWriteSkipReason | None
    skip_detail: str | None
    gradient_mode: GradientMode
    pre_write_cosine_mean: float = 0.0
    post_write_cosine_mean: float = 0.0
    key_pairwise_cosine_mean: float = 0.0
    value_pairwise_cosine_mean: float = 0.0
    delta_pairwise_cosine_mean: float = 0.0


class FastTTTAdapter(nn.Module):  # type: ignore[misc]
    """4096→768→768→4096 residual Adapter with an external per-video memory matrix."""

    def __init__(self, config: FastTTTConfig, memory: FastMemoryConfig) -> None:
        super().__init__()
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
            # per-seed lottery whose shared DC offset pinned sum(eta) to the budget --
            # the delta rule's annihilation point -- on some seeds and not others.
            # W_out still trains normally; only its init is pinned.
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
        mask = self._normalize_valid_mask(visual_embeddings, valid_mask)
        raw_runtime_states = fast_state if fast_state is not None else self._active_fast_states
        runtime_states = self._normalize_runtime_states(raw_runtime_states)
        detach_slow = False
        if runtime_states is not None:
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
        # invariant).  The exact ``context_term`` from above is re-added rather
        # than recomputed: dropping it would sever the only Bank->write-key
        # gradient channel.
        memory_keys = _centered_over_valid_tokens(base, mask) + context_term
        core_context = (
            torch.autocast(device_type=visual_embeddings.device.type, enabled=False)
            if visual_embeddings.device.type in {"cpu", "cuda"}
            else nullcontext()
        )
        with core_context:
            projected = projected.float()
            memory_keys = memory_keys.float()
            hidden = F.linear(projected, w0_1_value.float(), None)
            core = F.linear(F.silu(hidden), w0_2_value.float(), None)
            if runtime_states is not None:
                memory_stack = torch.stack([state.m for state in runtime_states])
                readout = alpha_value.float() * torch.bmm(
                    memory_keys,
                    memory_stack.transpose(1, 2),
                )
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
        self.last_audit = FastTTTForwardAudit(
            write_versions=write_versions,
            write_counts=write_counts,
            valid_token_counts=tuple(int(row.sum().item()) for row in mask),
            used_runtime_state=runtime_states is not None,
            bank_record_counts=tuple(int(value.item()) for value in bank_record_counts),
            readout_share_norms=(0.0,) * visual_embeddings.shape[0],
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

        slots = spatial.slots
        slot_mask = spatial.slot_valid_mask
        confidence = spatial.slot_confidence
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
            # sum(eta) <= eta_chunk_budget is the contraction bound the K=8
            # truncated meta-gradient relies on; the clamp keeps the rescale finite.
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

        states = (state,) if isinstance(state, FastMemoryState) else tuple(state)
        freeze_module = not states[0].differentiable
        previous_requires_grad = tuple(parameter.requires_grad for parameter in self.parameters())
        if freeze_module:
            # Functional, not audit: online binding must not build a module graph.
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

        self._active_associative_context = context
        try:
            yield self
        finally:
            self._active_associative_context = None

    def consume_associative_intermediates(self) -> AssociativeTTTIntermediates | None:
        """Take and clear the ephemeral tensors captured by the latest visual call."""

        value = self._last_associative_intermediates
        self._last_associative_intermediates = None
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

        ``memory_alpha`` and ``p_context`` are deliberately excluded: they gate the
        read path and would keep looking healthy across a dead write path.
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
        return valid_mask

    @staticmethod
    def _normalize_runtime_states(
        state: FastMemoryState | Sequence[FastMemoryState] | None,
    ) -> tuple[FastMemoryState, ...] | None:
        if state is None:
            return None
        return (state,) if isinstance(state, FastMemoryState) else tuple(state)

    @staticmethod
    def _online_value(value: Tensor | None, detach: bool) -> Tensor | None:
        if value is None:
            return None
        return value.detach() if detach else value


def make_query_proxy_fast_state(state: FastMemoryState) -> FastMemoryState:
    """Create an isolated leaf proxy with the exact numeric value of differentiable ``M``.

    Query-local backward may consume and release this proxy graph immediately.  A later
    deferred VJP explicitly links the captured proxy gradient back to the authoritative
    differentiable memory state, so this helper must not retain any edge to the Support
    graph: the detach -> clone -> requires_grad_ chain below is that isolation.
    """

    proxy = state.m.detach().clone().requires_grad_(True)
    return FastMemoryState(
        m=proxy,
        write_version=state.write_version,
        write_count=state.write_count,
        skip_count=state.skip_count,
        differentiable=True,
    )


def deferred_fast_vjp_loss(
    authoritative_states: Sequence[FastMemoryState],
    proxy_gradients: Sequence[Tensor],
) -> Tensor:
    """Return a numerically-zero scalar whose gradient injects a deferred memory VJP."""

    states = tuple(authoritative_states)
    gradients = tuple(proxy_gradients)
    authoritative = tuple(value for state in states for value in state.fast_parameters)
    terms: list[Tensor] = []
    for value, gradient in zip(authoritative, gradients, strict=True):
        accumulation_dtype = torch.promote_types(value.dtype, gradient.dtype)
        terms.append(
            (
                value.to(dtype=accumulation_dtype)
                * gradient.to(dtype=accumulation_dtype)
            ).sum()
        )
    link = torch.stack(terms).sum()
    # Preserve only d<link>/dM.  The arbitrary dot-product value is not part of L_total.
    return link - link.detach()


def build_fast_ttt_adapter(config: ProjectConfig) -> FastTTTAdapter:
    return FastTTTAdapter(config.fast_ttt, config.fast_memory)


def build_fast_associative_context(
    query_target: Tensor,
    bank_view: StateBankView,
) -> FastAssociativeContext:
    """Pool every pre-write present+valid semantic record with parameter-free attention."""

    embeddings = bank_view.embeddings.to(
        device=query_target.device,
        dtype=query_target.dtype,
    ).detach()
    valid = (
        bank_view.present_mask & bank_view.record_valid_mask
    ).to(device=query_target.device)
    counts = valid.sum(dim=1, dtype=torch.int64)
    if embeddings.shape[1] == 0:
        pooled = torch.zeros_like(query_target)
    else:
        scores = torch.einsum("bd,bnd->bn", query_target, embeddings)
        scores = scores / math.sqrt(BANK_EMBEDDING_DIM)
        # Empty-row correction: a fully-masked row would softmax over -inf and
        # produce NaN, so its scores are replaced by zeros and the weight sum is
        # floored at 1.0, which yields an exactly-zero pool instead.
        row_nonempty = valid.any(dim=1, keepdim=True)
        safe_scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        safe_scores = torch.where(row_nonempty, safe_scores, torch.zeros_like(safe_scores))
        weights = torch.softmax(safe_scores, dim=1) * valid.to(dtype=scores.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = torch.einsum("bn,bnd->bd", weights, embeddings)
    combined = query_target + pooled
    return FastAssociativeContext(
        combined_query=combined,
        bank_record_counts=counts,
        bank_versions=bank_view.bank_versions,
    )


def apply_memory_write(
    *,
    fast_state: FastMemoryState,
    batch: MemoryWriteBatch,
    row: int,
) -> MemoryWriteResult:
    """Return one new memory generation without mutating the current chunk state."""

    gradient_mode = (
        GradientMode.META_LINEAR_RECURRENCE
        if fast_state.differentiable
        else GradientMode.ONLINE_LEAF
    )
    mask = batch.slot_mask[row]
    slot_count = int(mask.sum().item())
    if slot_count == 0:
        return _skip_result(
            fast_state,
            reason=MemoryWriteSkipReason.NO_VALID_SLOT,
            detail="chunk_produced_no_valid_slot",
            gradient_mode=gradient_mode,
        )
    keys = batch.keys[row]
    values = batch.values[row]
    etas = batch.etas[row]
    row_payload_finite = bool(
        torch.isfinite(keys).all()
        and torch.isfinite(values).all()
        and torch.isfinite(etas).all()
        and torch.isfinite(batch.beta).all()
    )
    if not row_payload_finite:
        return _skip_result(
            fast_state,
            reason=MemoryWriteSkipReason.NONFINITE_KEY_VALUE,
            detail="write_payload_is_not_finite",
            gradient_mode=gradient_mode,
        )

    write_context = (
        torch.enable_grad() if fast_state.differentiable else torch.no_grad()
    )
    with write_context:
        recall = keys @ fast_state.m.transpose(0, 1)
        delta = values - recall
        update = torch.einsum("s,sd,se->de", etas, delta, keys)
        next_m = (1.0 - batch.beta) * fast_state.m + update
    if not bool(torch.isfinite(next_m.detach()).all()):
        return _skip_result(
            fast_state,
            reason=MemoryWriteSkipReason.NONFINITE_KEY_VALUE,
            detail="memory_after_write_is_not_finite",
            gradient_mode=gradient_mode,
        )
    with torch.no_grad():
        write_norm = float(
            torch.linalg.matrix_norm(next_m.detach() - fast_state.m.detach()).cpu().item()
        )
        memory_norm = float(torch.linalg.matrix_norm(next_m.detach()).cpu().item())
        eta_sum = float(etas.detach().sum().cpu().item())
    next_state = FastMemoryState(
        m=_next_memory(next_m, differentiable=fast_state.differentiable),
        write_version=fast_state.write_version + 1,
        write_count=fast_state.write_count + 1,
        skip_count=fast_state.skip_count,
        differentiable=fast_state.differentiable,
    )
    return MemoryWriteResult(
        fast_state=next_state,
        did_write=True,
        slots_written=slot_count,
        eta_sum=eta_sum,
        eta_renormalized=batch.eta_renormalized[row],
        write_norm=write_norm,
        memory_norm=memory_norm,
        skip_reason=None,
        skip_detail=None,
        gradient_mode=gradient_mode,
    )


def apply_memory_writes(
    *,
    fast_states: Sequence[FastMemoryState],
    batch: MemoryWriteBatch,
) -> tuple[MemoryWriteResult, ...]:
    """Apply independent row writes to a storage-isolated batch of video states."""

    states = tuple(fast_states)
    return tuple(
        apply_memory_write(fast_state=state, batch=batch, row=row)
        for row, state in enumerate(states)
    )


def truncate_memory_state(state: FastMemoryState) -> FastMemoryState:
    """Cut the K-step inner graph at a segment boundary, preserving values bitwise.

    Unlike the retired straight-through re-anchor there is no W0 ancestry to
    restore: the memory is zero-initialized per video, so the truncated leaf is
    the complete authoritative value.
    """

    next_m = state.m.detach().clone().requires_grad_(True)
    return FastMemoryState(
        m=next_m,
        write_version=state.write_version,
        write_count=state.write_count,
        skip_count=state.skip_count,
        differentiable=True,
    )


def truncate_memory_states(
    states: Sequence[FastMemoryState],
) -> tuple[FastMemoryState, ...]:
    """Truncate one batched trajectory's memories at a K=8 segment boundary."""

    return tuple(truncate_memory_state(state) for state in states)


def _skip_result(
    fast_state: FastMemoryState,
    *,
    reason: MemoryWriteSkipReason,
    detail: str,
    gradient_mode: GradientMode,
) -> MemoryWriteResult:
    next_state = FastMemoryState(
        m=_next_memory(fast_state.m, differentiable=fast_state.differentiable),
        write_version=fast_state.write_version,
        write_count=fast_state.write_count,
        skip_count=fast_state.skip_count + 1,
        differentiable=fast_state.differentiable,
    )
    with torch.no_grad():
        memory_norm = (
            0.0
            if fast_state.m.device.type == "meta"
            else float(torch.linalg.matrix_norm(fast_state.m.detach()).cpu().item())
        )
    return MemoryWriteResult(
        fast_state=next_state,
        did_write=False,
        slots_written=0,
        eta_sum=0.0,
        eta_renormalized=False,
        write_norm=0.0,
        memory_norm=memory_norm,
        skip_reason=reason,
        skip_detail=detail,
        gradient_mode=gradient_mode,
    )


def _next_memory(value: Tensor, *, differentiable: bool) -> Tensor:
    if differentiable:
        return value.clone()
    return value.detach().clone().requires_grad_(True)


def _smooth_normalize(value: Tensor) -> Tensor:
    """Normalize with a finite first/second derivative at the zero vector."""

    inverse_norm = torch.rsqrt(
        value.square().sum(dim=-1, keepdim=True) + _NORMALIZE_EPSILON**2
    )
    return value * inverse_norm


def _centered_over_valid_slots(slots: Tensor, slot_mask: Tensor) -> Tensor:
    """Remove the component every slot of a chunk shares, per batch row.

    Load-bearing capacity fix: the slots are the common ancestor of both write
    halves (keys via the probe attention, values via ``W_v``) and were measured
    at pairwise cosine 0.999, which pins ``rank(update)`` at 1.  Both the norm
    restoration (``gain``) and its ``clamp_max`` are corrections, not checks:
    centering alone shrinks the probe input ~150x and flattens the softmax to
    near-uniform, and the cap stops a slot sitting numerically on the mean from
    having its rounding noise amplified to full scale.  Rows with fewer than two
    valid slots pass through unchanged.
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

    The shared token mean is the remaining ceiling on realized write capacity
    (measured pairwise token cosine 0.46-0.50 caps the key participation ratio at
    ~4.2 of 32).  Only the visual component is centered; the caller re-adds the
    per-chunk ``p_context`` broadcast, which is the sole cross-chunk key-variation
    source and the sole Bank->write-key gradient channel.  No norm restoration is
    needed here: subtracting a per-chunk constant shifts every score row uniformly
    and leaves token selection provably unchanged.  Rows with fewer than two valid
    tokens pass through unchanged.
    """

    mask = valid_mask.unsqueeze(-1).to(dtype=tokens.dtype)
    counts = mask.sum(dim=1, keepdim=True)
    mean = (tokens * mask).sum(dim=1, keepdim=True) / counts.clamp_min(1.0)
    centerable = counts >= 2.0
    return torch.where(centerable, tokens - mean, tokens)
