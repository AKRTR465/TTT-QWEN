"""Expose the Qwen3-VL Main Merger boundary without copying upstream internals.

Inputs: causal Qwen video patches, grid metadata, a local checkpoint, and an optional adapter.
Outputs: padded per-video Main Merger features plus untouched DeepStack features and metadata.
Forbidden: image-path changes, DeepStack adaptation, State Bank access, online SGD, or copied Qwen
vision/LLM forward code.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Protocol, Self, cast

import torch
import transformers
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

from ttt_svcbench_qwen.config import ProjectConfig


class QwenFeatureOwner(Protocol):
    config: object
    visual: object
    language_model: object

    def get_video_features(
        self,
        pixel_values_videos: Tensor,
        video_grid_thw: Tensor | None = None,
    ) -> tuple[Sequence[Tensor], Sequence[Tensor]]: ...


@dataclass(frozen=True, slots=True)
class VideoBatch:
    """One causal video batch before the Qwen vision tower."""

    pixel_values_videos: Tensor
    video_grid_thw: Tensor
    timestamps: Tensor
    query_time: Tensor
    valid_mask: Tensor
    video_ids: tuple[str, ...]
    trajectory_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        pixels = self.pixel_values_videos
        if pixels.ndim != 2 or pixels.shape[-1] != 1536 or not torch.is_floating_point(pixels):
            raise ValueError("pixel_values_videos must be packed floating [sum(N_patch), 1536]")
        if self.video_grid_thw.ndim != 2 or self.video_grid_thw.shape[1] != 3:
            raise ValueError("video_grid_thw must be [B, 3]")
        batch_size = self.video_grid_thw.shape[0]
        if batch_size == 0:
            raise ValueError("video_grid_thw must contain at least one video")
        if self.video_grid_thw.dtype not in (torch.int32, torch.int64):
            raise TypeError("video_grid_thw must use an integer dtype")
        if bool(torch.any(self.video_grid_thw <= 0)):
            raise ValueError("video_grid_thw entries must be positive")
        if pixels.shape[0] != sum(self.patch_counts):
            raise ValueError("packed patch count must equal sum(prod(video_grid_thw))")
        if self.timestamps.ndim != 2 or self.timestamps.shape[0] != batch_size:
            raise ValueError("timestamps must be [B, T]")
        if not torch.is_floating_point(self.timestamps):
            raise TypeError("timestamps must use a floating dtype")
        if self.query_time.shape != (batch_size,) or not torch.is_floating_point(self.query_time):
            raise ValueError("query_time must be floating [B]")
        if self.valid_mask.shape != self.timestamps.shape or self.valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be bool [B, T]")
        if len(self.video_ids) != batch_size or len(self.trajectory_ids) != batch_size:
            raise ValueError("video_ids and trajectory_ids must contain one value per batch item")
        if not all(self.video_ids) or not all(self.trajectory_ids):
            raise ValueError("video_id and trajectory_id must be non-empty")

    @property
    def patch_counts(self) -> tuple[int, ...]:
        return tuple(int(value) for value in torch.prod(self.video_grid_thw, dim=1).tolist())

    @property
    def patch_offsets(self) -> tuple[int, ...]:
        offsets = [0]
        for count in self.patch_counts:
            offsets.append(offsets[-1] + count)
        return tuple(offsets)


@dataclass(frozen=True, slots=True)
class MergedVideoMetadata:
    """Packed-to-batch mapping at the Main Merger output."""

    video_grid_thw: Tensor
    merged_grid_thw: Tensor
    spatial_merge_size: int
    token_counts: tuple[int, ...]
    token_offsets: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.video_grid_thw.ndim != 2 or self.video_grid_thw.shape[1] != 3:
            raise ValueError("video_grid_thw must be [B, 3]")
        batch_size = self.video_grid_thw.shape[0]
        if batch_size == 0:
            raise ValueError("video_grid_thw must contain at least one video")
        if self.merged_grid_thw.shape != self.video_grid_thw.shape:
            raise ValueError("merged_grid_thw must match video_grid_thw")
        if self.video_grid_thw.dtype not in (torch.int32, torch.int64):
            raise TypeError("video grids must use an integer dtype")
        if self.merged_grid_thw.dtype != self.video_grid_thw.dtype:
            raise TypeError("raw and merged grids must use the same dtype")
        if self.spatial_merge_size <= 0:
            raise ValueError("spatial_merge_size must be positive")
        if bool(torch.any(self.video_grid_thw <= 0)) or bool(torch.any(self.merged_grid_thw <= 0)):
            raise ValueError("raw and merged grids must be positive")
        if not torch.equal(self.video_grid_thw[:, 0], self.merged_grid_thw[:, 0]):
            raise ValueError("Main Merger must preserve the temporal grid")
        if bool(torch.any(self.video_grid_thw[:, 1:] % self.spatial_merge_size != 0)):
            raise ValueError("raw spatial grids must be divisible by spatial_merge_size")
        expected_merged = self.video_grid_thw.clone()
        expected_merged[:, 1:] //= self.spatial_merge_size
        if not torch.equal(self.merged_grid_thw, expected_merged):
            raise ValueError("merged_grid_thw must divide only H/W by spatial_merge_size")
        computed_counts = tuple(
            int(value) for value in torch.prod(self.merged_grid_thw, dim=1).tolist()
        )
        if computed_counts != self.token_counts:
            raise ValueError("token_counts must equal prod(merged_grid_thw)")
        if len(self.token_counts) != batch_size:
            raise ValueError("token_counts must contain one value per video")
        if len(self.token_offsets) != batch_size + 1 or self.token_offsets[0] != 0:
            raise ValueError("token_offsets must be a B+1 prefix sum starting at zero")
        if any(count <= 0 for count in self.token_counts):
            raise ValueError("every video must contain at least one merged token")
        expected_offsets = [0]
        for count in self.token_counts:
            expected_offsets.append(expected_offsets[-1] + count)
        if tuple(expected_offsets) != self.token_offsets:
            raise ValueError("token_offsets must be the prefix sum of token_counts")


@dataclass(frozen=True, slots=True)
class QwenVisualOutput:
    """Padded Main Merger view plus original packed DeepStack tensors."""

    main_visual_embeddings: Tensor
    deepstack_features: tuple[Tensor, Tensor, Tensor]
    visual_valid_mask: Tensor
    metadata: MergedVideoMetadata

    def __post_init__(self) -> None:
        main = self.main_visual_embeddings
        batch_size = len(self.metadata.token_counts)
        if (
            main.ndim != 3
            or main.shape[0] != batch_size
            or main.shape[-1] <= 0
            or not torch.is_floating_point(main)
        ):
            raise ValueError("main_visual_embeddings must be floating [B, N_max, D]")
        if main.shape[1] != max(self.metadata.token_counts):
            raise ValueError("Main padding width must equal max(token_counts)")
        if self.visual_valid_mask.shape != main.shape[:2]:
            raise ValueError("visual_valid_mask must be [B, N_max]")
        if self.visual_valid_mask.dtype != torch.bool:
            raise TypeError("visual_valid_mask must use bool dtype")
        positions = torch.arange(main.shape[1], device=main.device).unsqueeze(0)
        counts = torch.tensor(
            self.metadata.token_counts,
            dtype=torch.int64,
            device=main.device,
        ).unsqueeze(1)
        if not torch.equal(self.visual_valid_mask, positions < counts):
            raise ValueError("visual_valid_mask must be a left-aligned token-count mask")
        if len(self.deepstack_features) != 3:
            raise ValueError("deepstack_features must contain exactly three packed tensors")
        for feature in self.deepstack_features:
            if (
                feature.shape != (self.metadata.token_offsets[-1], main.shape[-1])
                or feature.dtype != main.dtype
                or feature.device != main.device
            ):
                raise ValueError(
                    "each DeepStack feature must be packed [sum(N_i), D] with Main dtype/device"
                )

    @property
    def video_grid_thw(self) -> Tensor:
        return self.metadata.video_grid_thw

    @property
    def merged_grid_thw(self) -> Tensor:
        return self.metadata.merged_grid_thw

    @property
    def token_offsets(self) -> tuple[int, ...]:
        return self.metadata.token_offsets

    def packed_main_visual_embeddings(self) -> Tensor:
        return self.main_visual_embeddings[self.visual_valid_mask]

    def packed_deepstack_feature(self, index: int) -> Tensor:
        return self.deepstack_features[index]

    def padded_deepstack_feature(self, index: int) -> Tensor:
        splits = torch.split(self.deepstack_features[index], self.metadata.token_counts)
        return pad_sequence(splits, batch_first=True)

    def split_main_visual_embeddings(self) -> tuple[Tensor, ...]:
        return tuple(torch.split(self.packed_main_visual_embeddings(), self.metadata.token_counts))


@dataclass(frozen=True, slots=True, eq=False)
class PreparedVideoFeatures:
    """Actual adapted Main features plus untouched DeepStack tensors consumed by Qwen.

    The tensors intentionally keep their autograd graph and object identity. A prepared value is a
    one-prefill capability, not a persistent cache or detached runtime snapshot.
    """

    main_features: tuple[Tensor, ...]
    deepstack_features: tuple[Tensor, Tensor, Tensor]
    metadata: MergedVideoMetadata

    def __post_init__(self) -> None:
        hidden_size = self.main_features[0].shape[-1] if self.main_features else 0
        if hidden_size <= 0:
            raise ValueError("prepared Main features must contain at least one hidden dimension")
        _validate_feature_splits(
            self.main_features,
            self.metadata.token_counts,
            hidden_size,
            "prepared Main",
        )
        if len(self.deepstack_features) != 3:
            raise ValueError("prepared DeepStack features must contain exactly three tensors")
        first = self.main_features[0]
        for index, feature in enumerate(self.deepstack_features):
            if feature.shape != (self.metadata.token_offsets[-1], hidden_size):
                raise ValueError(f"prepared DeepStack feature {index} has an invalid packed shape")
            if feature.dtype != first.dtype or feature.device != first.device:
                raise ValueError(
                    "prepared DeepStack dtype/device must match prepared Main features"
                )

    def validate_request(self, pixel_values_videos: Tensor, video_grid_thw: Tensor) -> None:
        """Reject a provider request that is not the exact video geometry used for preparation."""

        if video_grid_thw.ndim != 2 or video_grid_thw.shape[1] != 3:
            raise ValueError("precomputed provider video_grid_thw must be [B, 3]")
        if video_grid_thw.dtype not in (torch.int32, torch.int64):
            raise TypeError("precomputed provider video_grid_thw must use an integer dtype")
        expected_grid = self.metadata.video_grid_thw.detach().to(device="cpu")
        requested_grid = video_grid_thw.detach().to(device="cpu")
        if not torch.equal(requested_grid, expected_grid):
            raise ValueError("precomputed provider video_grid_thw does not match prepared features")
        if pixel_values_videos.ndim != 2 or not torch.is_floating_point(pixel_values_videos):
            raise ValueError(
                "precomputed provider pixels must be packed floating [sum(N_patch), D]"
            )
        expected_patches = sum(
            int(value) for value in torch.prod(video_grid_thw.detach().to("cpu"), dim=1).tolist()
        )
        if pixel_values_videos.shape[0] != expected_patches:
            raise ValueError(
                "precomputed provider packed patch count does not match video_grid_thw"
            )


@dataclass(frozen=True, slots=True)
class CurrentChunkVisualTokenAudit:
    """Evidence that one Qwen prefill consumes exactly one current video chunk.

    Token counts are deliberately *not* required to be constant across chunks.  Spatial
    resolution/aspect ratio and the number of real frames may therefore change the count, while
    the continuation capability remains a single :class:`PreparedVideoFeatures` object rather
    than a container of historical chunk features.
    """

    batch_size: int
    raw_patch_counts: tuple[int, ...]
    merged_token_counts: tuple[int, ...]
    history_feature_set_count: int
    dynamic_token_count_allowed: bool

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("current-chunk visual audit requires a non-empty batch")
        if (
            len(self.raw_patch_counts) != self.batch_size
            or len(self.merged_token_counts) != self.batch_size
        ):
            raise ValueError("current-chunk visual audit counts must align to the batch")
        if any(value <= 0 for value in (*self.raw_patch_counts, *self.merged_token_counts)):
            raise ValueError("current-chunk visual token counts must be positive")
        if self.history_feature_set_count != 0:
            raise ValueError("historical chunk visual features may not enter a Qwen prefill")
        if not self.dynamic_token_count_allowed:
            raise ValueError("production current-chunk visual counts are intentionally dynamic")


def audit_current_chunk_visual_tokens(
    prepared: PreparedVideoFeatures,
    pixel_values_videos: Tensor,
    video_grid_thw: Tensor,
) -> CurrentChunkVisualTokenAudit:
    """Validate the one-current-chunk Qwen continuation boundary.

    The strict ``PreparedVideoFeatures`` type is important here: callers cannot pass a list or
    tuple of per-chunk capabilities and silently concatenate their visual tokens.  The exact
    geometry must also match the pixels of the current request.  No fixed merged-token target is
    imposed.
    """

    if not isinstance(prepared, PreparedVideoFeatures):
        raise TypeError(
            "Qwen prefill requires one PreparedVideoFeatures current chunk, not a history container"
        )
    prepared.validate_request(pixel_values_videos, video_grid_thw)
    raw_patch_counts = tuple(
        int(value) for value in torch.prod(video_grid_thw.detach().to("cpu"), dim=1).tolist()
    )
    return CurrentChunkVisualTokenAudit(
        batch_size=len(prepared.metadata.token_counts),
        raw_patch_counts=raw_patch_counts,
        merged_token_counts=prepared.metadata.token_counts,
        history_feature_set_count=0,
        dynamic_token_count_allowed=True,
    )


@dataclass(frozen=True, slots=True, eq=False)
class StateEmbeddingPayload:
    """Packed State embeddings to scatter once into one exact Qwen prefill sequence.

    ``state_embeddings`` is ``[N_state, H]`` in the row-major order selected by
    ``state_position_mask``. Expected IDs and the mask are immutable audit snapshots; State
    embeddings retain their graph so answer losses can reach the State Resampler.
    """

    expected_input_ids: Tensor
    state_position_mask: Tensor
    state_embeddings: Tensor

    def __post_init__(self) -> None:
        input_ids = self.expected_input_ids
        mask = self.state_position_mask
        embeddings = self.state_embeddings
        if (
            input_ids.ndim != 2
            or input_ids.shape[0] <= 0
            or input_ids.shape[1] <= 1
            or input_ids.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("expected_input_ids must be non-empty integer [B, L>1]")
        if mask.shape != input_ids.shape or mask.dtype != torch.bool:
            raise ValueError("state_position_mask must be bool with the expected input_ids shape")
        if mask.device != input_ids.device:
            raise ValueError("state_position_mask must share the expected input_ids device")
        if (
            embeddings.ndim != 2
            or embeddings.shape[1] <= 0
            or not torch.is_floating_point(embeddings)
        ):
            raise ValueError("state_embeddings must be floating [N_state, H]")
        if embeddings.shape[0] != int(mask.sum().item()):
            raise ValueError("state_embeddings rows must equal the State mask population")
        if embeddings.device != input_ids.device:
            raise ValueError("state_embeddings must share the expected input_ids device")
        if not bool(torch.isfinite(embeddings).all()):
            raise ValueError("state_embeddings must be finite")
        object.__setattr__(self, "expected_input_ids", input_ids.detach().clone())
        object.__setattr__(self, "state_position_mask", mask.detach().clone())


class QwenVideoFeatureBoundary(nn.Module):  # type: ignore[misc]
    """Adapt only packed Main Merger features and preserve upstream DeepStack objects."""

    def __init__(
        self,
        config: ProjectConfig,
        adapter: nn.Module | None = None,
        *,
        adapter_enabled: bool = False,
    ) -> None:
        super().__init__()
        if adapter_enabled and adapter is None:
            raise ValueError("adapter_enabled requires an adapter module")
        self._merge_size = config.model.vision.spatial_merge_size
        self._output_size = config.model.vision.output_size
        self._deepstack_count = len(config.model.vision.deepstack_visual_indexes)
        self.adapter = adapter
        self.adapter_enabled = adapter_enabled
        self.last_output: QwenVisualOutput | None = None
        self.last_prepared: PreparedVideoFeatures | None = None

    def set_adapter_enabled(self, enabled: bool) -> None:
        if enabled and self.adapter is None:
            raise ValueError("cannot enable a missing adapter module")
        self.adapter_enabled = enabled

    def intercept_features(
        self,
        main_features: Sequence[Tensor],
        deepstack_features: Sequence[Tensor],
        video_grid_thw: Tensor,
    ) -> tuple[Sequence[Tensor], Sequence[Tensor]]:
        output = self._capture(main_features, deepstack_features, video_grid_thw)
        self.last_output = output
        if not self.adapter_enabled:
            self.last_prepared = PreparedVideoFeatures(
                main_features=tuple(main_features),
                deepstack_features=output.deepstack_features,
                metadata=output.metadata,
            )
            return main_features, deepstack_features
        if self.adapter is None:
            raise RuntimeError("adapter disappeared while adapter_enabled is true")
        adapted = cast(
            Tensor,
            self.adapter(
                output.main_visual_embeddings,
                output.visual_valid_mask,
                output.metadata,
            ),
        )
        if adapted.shape != output.main_visual_embeddings.shape:
            raise ValueError("video adapter must preserve [B, N_max, 4096] shape")
        if (
            adapted.dtype != output.main_visual_embeddings.dtype
            or adapted.device != output.main_visual_embeddings.device
        ):
            raise ValueError("video adapter must preserve Main Merger dtype/device")
        packed = adapted[output.visual_valid_mask]
        adapted_splits = tuple(torch.split(packed, output.metadata.token_counts))
        self.last_prepared = PreparedVideoFeatures(
            main_features=adapted_splits,
            deepstack_features=output.deepstack_features,
            metadata=output.metadata,
        )
        return adapted_splits, deepstack_features

    def _capture(
        self,
        main_features: Sequence[Tensor],
        deepstack_features: Sequence[Tensor],
        video_grid_thw: Tensor,
    ) -> QwenVisualOutput:
        metadata = _build_merged_metadata(video_grid_thw, self._merge_size)
        main_splits = tuple(main_features)
        if len(main_splits) != len(metadata.token_counts):
            raise ValueError("Qwen returned a different number of videos than video_grid_thw")
        _validate_feature_splits(
            main_splits,
            metadata.token_counts,
            self._output_size,
            "Main Merger",
        )
        if len(deepstack_features) != self._deepstack_count:
            raise ValueError("Qwen returned an unexpected number of DeepStack feature groups")
        main_padded = pad_sequence(main_splits, batch_first=True)
        mask = _left_aligned_mask(metadata.token_counts, main_padded.shape[1], main_padded.device)
        raw_deepstack: list[Tensor] = []
        for index, feature in enumerate(deepstack_features):
            if feature.ndim != 2 or feature.shape != (
                metadata.token_offsets[-1],
                self._output_size,
            ):
                raise ValueError(f"DeepStack feature {index} has an invalid packed shape")
            if feature.dtype != main_padded.dtype or feature.device != main_padded.device:
                raise ValueError("DeepStack dtype/device must match Main Merger output")
            raw_deepstack.append(feature)
        fixed_deepstack = cast(tuple[Tensor, Tensor, Tensor], tuple(raw_deepstack))
        return QwenVisualOutput(
            main_visual_embeddings=main_padded,
            deepstack_features=fixed_deepstack,
            visual_valid_mask=mask,
            metadata=metadata,
        )


class Qwen3VLAdapter(nn.Module):  # type: ignore[misc]
    """Transparent model wrapper with a temporary video-feature interception surface."""

    def __init__(
        self,
        qwen_model: nn.Module,
        config: ProjectConfig,
        adapter: nn.Module | None = None,
        *,
        adapter_enabled: bool = False,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        self.qwen_model = qwen_model
        feature_owner = _resolve_feature_owner(qwen_model)
        assert_qwen_checkpoint_config(feature_owner.config, config)
        assert_qwen_runtime_structure(feature_owner, config)
        self._freeze_base = freeze_base
        if self._freeze_base:
            self.qwen_model.requires_grad_(False)
            self.qwen_model.eval()
        self.video_boundary = QwenVideoFeatureBoundary(
            config,
            adapter,
            adapter_enabled=adapter_enabled,
        )
        self._hook_active = False
        self._state_hook_active = False
        self._hook_lock = RLock()

    @property
    def last_visual_output(self) -> QwenVisualOutput | None:
        return self.video_boundary.last_output

    @property
    def last_prepared_video_features(self) -> PreparedVideoFeatures | None:
        """Return the actual post-Adapter Main features most recently handed to Qwen."""

        return self.video_boundary.last_prepared

    @property
    def feature_owner(self) -> QwenFeatureOwner:
        """Return the inner HF owner without registering it as a duplicate child module."""

        return _resolve_feature_owner(self.qwen_model)

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        if self._freeze_base:
            self.qwen_model.eval()
        return self

    def forward(
        self,
        *args: object,
        prepared_video_features: PreparedVideoFeatures | None = None,
        state_embedding_payload: StateEmbeddingPayload | None = None,
        **kwargs: object,
    ) -> object:
        with self._hook_lock:
            self._clear_captures()
            if state_embedding_payload is not None:
                self._validate_state_call(args, kwargs, state_embedding_payload)
            try:
                with (
                    self._patched_state_embeddings(state_embedding_payload),
                    self._patched_video_features(prepared_video_features),
                ):
                    return cast(object, self.qwen_model(*args, **kwargs))
            except Exception:
                self._clear_captures()
                raise

    def generate(
        self,
        *args: object,
        prepared_video_features: PreparedVideoFeatures | None = None,
        state_embedding_payload: StateEmbeddingPayload | None = None,
        **kwargs: object,
    ) -> object:
        generate_method = getattr(self.qwen_model, "generate", None)
        if not callable(generate_method):
            raise TypeError("wrapped Qwen model does not provide generate()")
        with self._hook_lock:
            self._clear_captures()
            if state_embedding_payload is not None:
                self._validate_state_call(args, kwargs, state_embedding_payload)
            if prepared_video_features is not None or state_embedding_payload is not None:
                self._assert_unexpanded_generation(kwargs)
            try:
                with (
                    self._patched_state_embeddings(state_embedding_payload),
                    self._patched_video_features(prepared_video_features),
                ):
                    return cast(object, generate_method(*args, **kwargs))
            except Exception:
                self._clear_captures()
                raise

    def get_video_features(
        self,
        pixel_values_videos: Tensor,
        video_grid_thw: Tensor,
    ) -> tuple[Sequence[Tensor], Sequence[Tensor]]:
        with self._hook_lock:
            if self._hook_active:
                raise RuntimeError(
                    "direct get_video_features cannot run while the forward hook is active"
                )
            self._clear_captures()
            try:
                main, deepstack = self.feature_owner.get_video_features(
                    pixel_values_videos,
                    video_grid_thw,
                )
                return self.video_boundary.intercept_features(
                    main,
                    deepstack,
                    video_grid_thw,
                )
            except Exception:
                self._clear_captures()
                raise

    @contextmanager
    def _patched_video_features(
        self,
        prepared: PreparedVideoFeatures | None = None,
    ) -> Iterator[None]:
        with self._hook_lock:
            if self._hook_active:
                raise RuntimeError("Qwen video feature hook is not re-entrant")
            owner = self.feature_owner
            had_instance_method = "get_video_features" in vars(owner)
            original_instance_method = vars(owner).get("get_video_features")
            original = owner.get_video_features
            prepared_consumed = False

            def intercepted(
                pixel_values_videos: Tensor,
                video_grid_thw: Tensor | None = None,
            ) -> tuple[Sequence[Tensor], Sequence[Tensor]]:
                nonlocal prepared_consumed
                if video_grid_thw is None:
                    raise ValueError("video_grid_thw is required for the State-TTT boundary")
                if prepared is not None:
                    if prepared_consumed:
                        raise RuntimeError(
                            "precomputed video features may be consumed only once per prefill"
                        )
                    prepared.validate_request(pixel_values_videos, video_grid_thw)
                    prepared_consumed = True
                    self.video_boundary.last_prepared = prepared
                    return prepared.main_features, prepared.deepstack_features
                main, deepstack = original(pixel_values_videos, video_grid_thw)
                return self.video_boundary.intercept_features(
                    main,
                    deepstack,
                    video_grid_thw,
                )

            self._hook_active = True
            owner.get_video_features = intercepted  # type: ignore[method-assign]
            try:
                yield
                if prepared is not None and not prepared_consumed:
                    raise RuntimeError(
                        "precomputed video features were not consumed exactly once during prefill"
                    )
            finally:
                if had_instance_method:
                    owner.get_video_features = original_instance_method  # type: ignore[method-assign,assignment]
                else:
                    delattr(owner, "get_video_features")
                self._hook_active = False

    @contextmanager
    def _patched_state_embeddings(
        self,
        payload: StateEmbeddingPayload | None = None,
    ) -> Iterator[None]:
        if payload is None:
            yield
            return
        with self._hook_lock:
            if self._state_hook_active:
                raise RuntimeError("Qwen State embedding hook is not re-entrant")
            get_embeddings = getattr(self.qwen_model, "get_input_embeddings", None)
            if not callable(get_embeddings):
                raise TypeError("wrapped Qwen model does not expose get_input_embeddings()")
            embedding_layer = get_embeddings()
            if not isinstance(embedding_layer, nn.Module):
                raise TypeError("Qwen input embedding owner must be an nn.Module")
            payload_consumed = False

            def scatter_state(
                _module: nn.Module,
                module_args: tuple[object, ...],
                output: object,
            ) -> object:
                nonlocal payload_consumed
                if not module_args or not isinstance(module_args[0], Tensor):
                    raise TypeError("Qwen input embedding hook requires Tensor input_ids")
                actual_ids = module_args[0]
                if not isinstance(output, Tensor):
                    raise TypeError("Qwen input embedding hook requires a Tensor output")
                if actual_ids.ndim == 2 and actual_ids.shape[1] == 1:
                    return output
                if payload_consumed:
                    raise RuntimeError(
                        "State embeddings were already consumed; decode must use one-token inputs"
                    )
                expected_ids = payload.expected_input_ids
                if (
                    actual_ids.shape != expected_ids.shape
                    or actual_ids.dtype != expected_ids.dtype
                    or actual_ids.device != expected_ids.device
                    or not torch.equal(actual_ids, expected_ids)
                ):
                    raise ValueError("Qwen prefill input_ids do not match StateEmbeddingPayload")
                if output.ndim != 3 or output.shape[:2] != actual_ids.shape:
                    raise ValueError("Qwen input embeddings must be [B, L, H]")
                if output.shape[-1] != payload.state_embeddings.shape[-1]:
                    raise ValueError("State embedding hidden size does not match Qwen embeddings")
                state_mask = payload.state_position_mask.to(device=output.device)
                expanded_mask = state_mask.unsqueeze(-1).expand_as(output)
                state_values = payload.state_embeddings.to(
                    device=output.device,
                    dtype=output.dtype,
                )
                scattered = output.masked_scatter(expanded_mask, state_values)
                payload_consumed = True
                return scattered

            self._state_hook_active = True
            handle = embedding_layer.register_forward_hook(scatter_state)
            try:
                yield
                if not payload_consumed:
                    raise RuntimeError(
                        "StateEmbeddingPayload was not consumed exactly once during prefill"
                    )
            finally:
                handle.remove()
                self._state_hook_active = False

    def _clear_captures(self) -> None:
        self.video_boundary.last_output = None
        self.video_boundary.last_prepared = None

    def _validate_state_call(
        self,
        args: Sequence[object],
        kwargs: Mapping[str, object],
        payload: StateEmbeddingPayload,
    ) -> None:
        if args:
            raise ValueError("StateEmbeddingPayload requires keyword input_ids")
        if kwargs.get("inputs_embeds") is not None:
            raise ValueError("StateEmbeddingPayload cannot be combined with inputs_embeds")
        input_ids = kwargs.get("input_ids")
        if not isinstance(input_ids, Tensor):
            raise ValueError("StateEmbeddingPayload requires Tensor input_ids")
        expected_ids = payload.expected_input_ids
        if (
            input_ids.shape != expected_ids.shape
            or input_ids.dtype != expected_ids.dtype
            or input_ids.device != expected_ids.device
            or not torch.equal(input_ids, expected_ids)
        ):
            raise ValueError("input_ids do not match StateEmbeddingPayload expected_input_ids")

    def _assert_unexpanded_generation(self, kwargs: Mapping[str, object]) -> None:
        generation_config = kwargs.get("generation_config")
        if generation_config is None:
            generation_config = getattr(self.qwen_model, "generation_config", None)
        for name in ("num_beams", "num_return_sequences"):
            value = kwargs.get(name)
            if value is None and generation_config is not None:
                value = getattr(generation_config, name, 1)
            if value is None:
                value = 1
            if value != 1:
                raise ValueError(
                    "precomputed video generation requires "
                    f"{name}=1; expanded generation is deferred"
                )


def assert_qwen_checkpoint_config(checkpoint_config: object, project: ProjectConfig) -> None:
    """Fail before weight execution when the loaded checkpoint violates the pinned P3 contract."""

    if transformers.__version__ != project.model.transformers_version:
        raise ValueError(
            "Transformers version mismatch: "
            f"expected {project.model.transformers_version}, got {transformers.__version__}"
        )
    vision = _required_attribute(checkpoint_config, "vision_config", "checkpoint")
    text = _required_attribute(checkpoint_config, "text_config", "checkpoint")
    expected_vision = project.model.vision
    expected_text = project.model.llm
    checks: tuple[tuple[str, object, object], ...] = (
        (
            "vision.depth",
            _required_attribute(vision, "depth", "vision_config"),
            expected_vision.depth,
        ),
        (
            "vision.hidden_size",
            _required_attribute(vision, "hidden_size", "vision_config"),
            expected_vision.hidden_size,
        ),
        (
            "vision.num_heads",
            _required_attribute(vision, "num_heads", "vision_config"),
            expected_vision.num_heads,
        ),
        ("vision.in_channels", _required_attribute(vision, "in_channels", "vision_config"), 3),
        (
            "vision.patch_size",
            _required_attribute(vision, "patch_size", "vision_config"),
            expected_vision.patch_size,
        ),
        (
            "vision.temporal_patch_size",
            _required_attribute(vision, "temporal_patch_size", "vision_config"),
            expected_vision.temporal_patch_size,
        ),
        (
            "vision.spatial_merge_size",
            _required_attribute(vision, "spatial_merge_size", "vision_config"),
            expected_vision.spatial_merge_size,
        ),
        (
            "vision.out_hidden_size",
            _required_attribute(vision, "out_hidden_size", "vision_config"),
            expected_vision.output_size,
        ),
        (
            "vision.deepstack_visual_indexes",
            tuple(
                cast(
                    Sequence[int],
                    _required_attribute(vision, "deepstack_visual_indexes", "vision_config"),
                )
            ),
            expected_vision.deepstack_visual_indexes,
        ),
        (
            "text.hidden_size",
            _required_attribute(text, "hidden_size", "text_config"),
            expected_text.hidden_size,
        ),
        (
            "text.num_hidden_layers",
            _required_attribute(text, "num_hidden_layers", "text_config"),
            expected_text.num_layers,
        ),
    )
    for path, actual, expected in checks:
        if actual != expected:
            raise ValueError(f"Qwen checkpoint {path} must be {expected!r}; got {actual!r}")


def assert_qwen_runtime_structure(owner: QwenFeatureOwner, project: ProjectConfig) -> None:
    """Verify the actual loaded modules expose the pinned merger and DeepStack structure."""

    vision_config = project.model.vision
    text_config = project.model.llm
    visual = _required_attribute(owner, "visual", "Qwen feature owner")
    language_model = _required_attribute(owner, "language_model", "Qwen feature owner")
    patch_embed = _required_attribute(visual, "patch_embed", "visual")
    patch_projection = _required_attribute(patch_embed, "proj", "visual.patch_embed")
    merger = _required_attribute(visual, "merger", "visual")
    merger_fc1 = _required_attribute(merger, "linear_fc1", "visual.merger")
    merger_fc2 = _required_attribute(merger, "linear_fc2", "visual.merger")
    blocks = _module_sequence(_required_attribute(visual, "blocks", "visual"), "visual.blocks")
    deepstack_mergers = _module_sequence(
        _required_attribute(visual, "deepstack_merger_list", "visual"),
        "visual.deepstack_merger_list",
    )
    decoder_layers = _module_sequence(
        _required_attribute(language_model, "layers", "language_model"),
        "language_model.layers",
    )
    runtime_checks: tuple[tuple[str, object, object], ...] = (
        (
            "visual.patch_embed.embed_dim",
            _required_attribute(patch_embed, "embed_dim", "patch_embed"),
            vision_config.hidden_size,
        ),
        (
            "visual.patch_embed.proj.kernel_size",
            tuple(
                cast(
                    Sequence[int],
                    _required_attribute(
                        patch_projection,
                        "kernel_size",
                        "visual.patch_embed.proj",
                    ),
                )
            ),
            (
                vision_config.temporal_patch_size,
                vision_config.patch_size,
                vision_config.patch_size,
            ),
        ),
        (
            "visual.patch_embed.proj.stride",
            tuple(
                cast(
                    Sequence[int],
                    _required_attribute(
                        patch_projection,
                        "stride",
                        "visual.patch_embed.proj",
                    ),
                )
            ),
            (
                vision_config.temporal_patch_size,
                vision_config.patch_size,
                vision_config.patch_size,
            ),
        ),
        (
            "visual.patch_embed.proj.in_channels",
            _required_attribute(
                patch_projection,
                "in_channels",
                "visual.patch_embed.proj",
            ),
            3,
        ),
        ("visual.blocks", len(blocks), vision_config.depth),
        (
            "visual.merger.hidden_size",
            _required_attribute(merger, "hidden_size", "visual.merger"),
            vision_config.hidden_size * vision_config.spatial_merge_size**2,
        ),
        (
            "visual.merger.linear_fc1.in_features",
            _required_attribute(merger_fc1, "in_features", "visual.merger.linear_fc1"),
            vision_config.hidden_size * vision_config.spatial_merge_size**2,
        ),
        (
            "visual.merger.linear_fc1.out_features",
            _required_attribute(merger_fc1, "out_features", "visual.merger.linear_fc1"),
            vision_config.hidden_size * vision_config.spatial_merge_size**2,
        ),
        (
            "visual.merger.linear_fc2.in_features",
            _required_attribute(merger_fc2, "in_features", "visual.merger.linear_fc2"),
            vision_config.hidden_size * vision_config.spatial_merge_size**2,
        ),
        (
            "visual.merger.linear_fc2.out_features",
            _required_attribute(merger_fc2, "out_features", "visual.merger.linear_fc2"),
            vision_config.output_size,
        ),
        (
            "visual.deepstack_visual_indexes",
            tuple(
                cast(
                    Sequence[int],
                    _required_attribute(visual, "deepstack_visual_indexes", "visual"),
                )
            ),
            vision_config.deepstack_visual_indexes,
        ),
        ("visual.deepstack_merger_list", len(deepstack_mergers), 3),
        ("language_model.layers", len(decoder_layers), text_config.num_layers),
    )
    for path, actual, expected in runtime_checks:
        if actual != expected:
            raise ValueError(f"Qwen runtime {path} must be {expected!r}; got {actual!r}")
    for index, deepstack_merger in enumerate(deepstack_mergers):
        fc2 = _required_attribute(
            deepstack_merger,
            "linear_fc2",
            f"visual.deepstack_merger_list[{index}]",
        )
        out_features = _required_attribute(
            fc2,
            "out_features",
            f"visual.deepstack_merger_list[{index}].linear_fc2",
        )
        if out_features != vision_config.output_size:
            raise ValueError(
                f"Qwen runtime DeepStack merger {index} output must be "
                f"{vision_config.output_size!r}; got {out_features!r}"
            )


def build_qwen_adapter(
    config: ProjectConfig | None = None,
    model_root: str | Path | None = None,
    *,
    adapter: nn.Module | None = None,
    adapter_enabled: bool = False,
    freeze_base: bool = True,
    device_map: str | Mapping[str, object] | None = "auto",
) -> Qwen3VLAdapter:
    """Load only a local pinned Qwen3-VL checkpoint and install the P3 boundary."""

    if config is None:
        raise ValueError("build_qwen_adapter requires a validated ProjectConfig")
    if model_root is None:
        raise ValueError("build_qwen_adapter requires a local model_root")
    root = Path(model_root)
    if not root.is_dir():
        raise FileNotFoundError(f"local Qwen model root does not exist: {root}")
    checkpoint_config = Qwen3VLConfig.from_pretrained(root, local_files_only=True)
    assert_qwen_checkpoint_config(checkpoint_config, config)
    kwargs: dict[str, object] = {
        "local_files_only": True,
        "dtype": "auto",
    }
    if device_map is not None:
        kwargs["device_map"] = device_map
    model = cast(
        nn.Module,
        Qwen3VLForConditionalGeneration.from_pretrained(root, **kwargs),
    )
    return Qwen3VLAdapter(
        model,
        config,
        adapter,
        adapter_enabled=adapter_enabled,
        freeze_base=freeze_base,
    )


def _resolve_feature_owner(model: nn.Module) -> QwenFeatureOwner:
    candidate = getattr(model, "model", model)
    if not callable(getattr(candidate, "get_video_features", None)):
        raise TypeError("Qwen model must expose get_video_features on itself or .model")
    return cast(QwenFeatureOwner, candidate)


def _build_merged_metadata(video_grid_thw: Tensor, merge_size: int) -> MergedVideoMetadata:
    if video_grid_thw.ndim != 2 or video_grid_thw.shape[1] != 3:
        raise ValueError("video_grid_thw must be [B, 3]")
    if video_grid_thw.dtype not in (torch.int32, torch.int64):
        raise TypeError("video_grid_thw must use an integer dtype")
    if bool(torch.any(video_grid_thw <= 0)):
        raise ValueError("video_grid_thw entries must be positive")
    if bool(torch.any(video_grid_thw[:, 1:] % merge_size != 0)):
        raise ValueError("video spatial grids must be divisible by spatial_merge_size")
    merged = video_grid_thw.clone()
    merged[:, 1:] = merged[:, 1:] // merge_size
    token_counts = tuple(int(value) for value in torch.prod(merged, dim=1).tolist())
    offsets = [0]
    for count in token_counts:
        offsets.append(offsets[-1] + count)
    return MergedVideoMetadata(
        video_grid_thw=video_grid_thw,
        merged_grid_thw=merged,
        spatial_merge_size=merge_size,
        token_counts=token_counts,
        token_offsets=tuple(offsets),
    )


def _validate_feature_splits(
    features: Sequence[Tensor],
    token_counts: Sequence[int],
    hidden_size: int,
    label: str,
) -> None:
    first = features[0] if features else None
    if first is None:
        raise ValueError(f"{label} returned no video features")
    for index, (feature, expected_tokens) in enumerate(zip(features, token_counts, strict=True)):
        if feature.ndim != 2 or feature.shape != (expected_tokens, hidden_size):
            raise ValueError(f"{label} video {index} has an invalid shape")
        if not torch.is_floating_point(feature):
            raise TypeError(f"{label} features must use a floating dtype")
        if feature.dtype != first.dtype or feature.device != first.device:
            raise ValueError(f"{label} feature splits must share dtype/device")


def _left_aligned_mask(token_counts: Sequence[int], width: int, device: torch.device) -> Tensor:
    positions = torch.arange(width, device=device).unsqueeze(0)
    counts = torch.tensor(tuple(token_counts), dtype=torch.int64, device=device).unsqueeze(1)
    return positions < counts


def _required_attribute(owner: object, name: str, label: str) -> object:
    if not hasattr(owner, name):
        raise ValueError(f"{label} is missing required attribute {name}")
    return getattr(owner, name)


def _module_sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, (list, tuple, nn.ModuleList)):
        raise ValueError(f"{label} must be a module sequence")
    return cast(Sequence[object], value)
