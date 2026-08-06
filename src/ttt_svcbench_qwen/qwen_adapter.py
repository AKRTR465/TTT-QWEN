"""Expose the Qwen3-VL Main Merger boundary without copying upstream internals.

Inputs: causal Qwen video patches, grid metadata, a local checkpoint, and an optional adapter.
Outputs: padded per-video Main Merger features plus untouched DeepStack features and metadata.
Forbidden: image-path changes, DeepStack adaptation, State Bank access, online SGD, or copied Qwen
vision/LLM forward code.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from typing import Protocol, Self, cast

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence

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
class MergedVideoMetadata:
    """Packed-to-batch mapping at the Main Merger output."""

    video_grid_thw: Tensor
    merged_grid_thw: Tensor
    spatial_merge_size: int
    token_counts: tuple[int, ...]
    token_offsets: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class QwenVisualOutput:
    """Padded Main Merger view plus original packed DeepStack tensors."""

    main_visual_embeddings: Tensor
    deepstack_features: tuple[Tensor, Tensor, Tensor]
    visual_valid_mask: Tensor
    metadata: MergedVideoMetadata

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

    def split_main_visual_embeddings(self) -> tuple[Tensor, ...]:
        return tuple(torch.split(self.packed_main_visual_embeddings(), self.metadata.token_counts))


@dataclass(frozen=True, slots=True)
class RawVisualChunk:
    """One pre-Adapter Main/DeepStack feature set split from a visual batch."""

    main_feature: Tensor
    deepstack_features: tuple[Tensor, Tensor, Tensor]
    metadata: MergedVideoMetadata
    source: object | None = None

    def as_batch(self) -> RawVideoFeatureBatch:
        valid = torch.ones(
            (1, self.main_feature.shape[0]),
            dtype=torch.bool,
            device=self.main_feature.device,
        )
        return RawVideoFeatureBatch(
            QwenVisualOutput(
                main_visual_embeddings=self.main_feature.unsqueeze(0),
                deepstack_features=self.deepstack_features,
                visual_valid_mask=valid,
                metadata=self.metadata,
            )
        )


@dataclass(frozen=True, slots=True)
class RawVideoFeatureBatch:
    """Pre-Adapter Main Merger/DeepStack output with exact grid and token offsets."""

    value: QwenVisualOutput

    def split(self) -> tuple[RawVisualChunk, ...]:
        main = self.value.split_main_visual_embeddings()
        rows: list[RawVisualChunk] = []
        for index, feature in enumerate(main):
            start = self.value.metadata.token_offsets[index]
            stop = self.value.metadata.token_offsets[index + 1]
            deepstack = tuple(value[start:stop] for value in self.value.deepstack_features)
            rows.append(
                RawVisualChunk(
                    main_feature=feature,
                    deepstack_features=cast(tuple[Tensor, Tensor, Tensor], deepstack),
                    metadata=_single_video_metadata(self.value.metadata, index),
                )
            )
        return tuple(rows)


@dataclass(frozen=True, slots=True, eq=False)
class PreparedVideoFeatures:
    """Actual adapted Main features plus untouched DeepStack tensors consumed by Qwen.

    The tensors intentionally keep their autograd graph and object identity. A prepared value is a
    one-prefill capability, not a persistent cache or detached runtime snapshot.
    """

    main_features: tuple[Tensor, ...]
    deepstack_features: tuple[Tensor, Tensor, Tensor]
    metadata: MergedVideoMetadata


@dataclass(frozen=True, slots=True, eq=False)
class PreparedVisualChunk:
    """One post-Adapter chunk ready for State modules and a single Qwen continuation."""

    value: QwenVisualOutput
    prepared_video_features: PreparedVideoFeatures
    source: object | None = None


def _single_video_metadata(
    metadata: MergedVideoMetadata,
    index: int,
) -> MergedVideoMetadata:
    count = metadata.token_counts[index]
    return MergedVideoMetadata(
        video_grid_thw=metadata.video_grid_thw[index : index + 1],
        merged_grid_thw=metadata.merged_grid_thw[index : index + 1],
        spatial_merge_size=metadata.spatial_merge_size,
        token_counts=(count,),
        token_offsets=(0, count),
    )


def _split_prepared_visual_batch(
    raw: QwenVisualOutput,
    prepared: PreparedVideoFeatures,
) -> tuple[PreparedVisualChunk, ...]:
    rows: list[PreparedVisualChunk] = []
    for index, main in enumerate(prepared.main_features):
        start = raw.metadata.token_offsets[index]
        stop = raw.metadata.token_offsets[index + 1]
        metadata = _single_video_metadata(raw.metadata, index)
        deepstack = cast(
            tuple[Tensor, Tensor, Tensor],
            tuple(feature[start:stop] for feature in prepared.deepstack_features),
        )
        row_prepared = PreparedVideoFeatures(
            main_features=(main,),
            deepstack_features=deepstack,
            metadata=metadata,
        )
        rows.append(
            PreparedVisualChunk(
                value=QwenVisualOutput(
                    main_visual_embeddings=main.unsqueeze(0),
                    deepstack_features=deepstack,
                    visual_valid_mask=torch.ones(
                        (1, main.shape[0]),
                        dtype=torch.bool,
                        device=main.device,
                    ),
                    metadata=metadata,
                ),
                prepared_video_features=row_prepared,
            )
        )
    return tuple(rows)


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
        self._merge_size = config.model.vision.spatial_merge_size
        self.adapter = adapter
        self.adapter_enabled = adapter_enabled
        self.last_output: QwenVisualOutput | None = None
        self.last_prepared: PreparedVideoFeatures | None = None

    def intercept_features(
        self,
        main_features: Sequence[Tensor],
        deepstack_features: Sequence[Tensor],
        video_grid_thw: Tensor,
    ) -> tuple[Sequence[Tensor], Sequence[Tensor]]:
        raw = self.capture_raw(main_features, deepstack_features, video_grid_thw)
        if not self.adapter_enabled:
            self.last_output = raw.value
            self.last_prepared = PreparedVideoFeatures(
                main_features=tuple(main_features),
                deepstack_features=raw.value.deepstack_features,
                metadata=raw.value.metadata,
            )
            return main_features, deepstack_features
        prepared = self._prepare_output(raw.value)
        return prepared.main_features, deepstack_features

    def capture_raw(
        self,
        main_features: Sequence[Tensor],
        deepstack_features: Sequence[Tensor],
        video_grid_thw: Tensor,
    ) -> RawVideoFeatureBatch:
        return RawVideoFeatureBatch(
            self._capture(main_features, deepstack_features, video_grid_thw)
        )

    def prepare_raw_batch(
        self,
        raw: RawVideoFeatureBatch,
    ) -> tuple[PreparedVisualChunk, ...]:
        prepared = self._prepare_output(raw.value)
        return _split_prepared_visual_batch(raw.value, prepared)

    def prepare_raw_chunk(self, raw: RawVisualChunk) -> PreparedVisualChunk:
        prepared = self.prepare_raw_batch(raw.as_batch())
        return prepared[0]

    def _prepare_output(self, output: QwenVisualOutput) -> PreparedVideoFeatures:
        self.last_output = output
        if not self.adapter_enabled:
            main = output.split_main_visual_embeddings()
        else:
            adapted = cast(
                Tensor,
                self.adapter(
                    output.main_visual_embeddings,
                    output.visual_valid_mask,
                    output.metadata,
                ),
            )
            packed = adapted[output.visual_valid_mask]
            main = tuple(torch.split(packed, output.metadata.token_counts))
        self.last_prepared = PreparedVideoFeatures(
            main_features=main,
            deepstack_features=output.deepstack_features,
            metadata=output.metadata,
        )
        return self.last_prepared

    def _capture(
        self,
        main_features: Sequence[Tensor],
        deepstack_features: Sequence[Tensor],
        video_grid_thw: Tensor,
    ) -> QwenVisualOutput:
        metadata = _build_merged_metadata(video_grid_thw, self._merge_size)
        main_splits = tuple(main_features)
        main_padded = pad_sequence(main_splits, batch_first=True)
        mask = _left_aligned_mask(metadata.token_counts, main_padded.shape[1], main_padded.device)
        fixed_deepstack = cast(tuple[Tensor, Tensor, Tensor], tuple(deepstack_features))
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
        self._freeze_base = freeze_base
        if self._freeze_base:
            self.qwen_model.requires_grad_(False)
            self.qwen_model.eval()
        self.video_boundary = QwenVideoFeatureBoundary(
            config,
            adapter,
            adapter_enabled=adapter_enabled,
        )
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
        state_embedding_payload: StateEmbeddingPayload | None = None,
        **kwargs: object,
    ) -> object:
        with self._hook_lock:
            self._clear_captures()
            try:
                with (
                    self._patched_state_embeddings(state_embedding_payload),
                    self._patched_video_features(),
                ):
                    return cast(object, self.qwen_model(*args, **kwargs))
            except Exception:
                self._clear_captures()
                raise

    def generate(
        self,
        *args: object,
        state_embedding_payload: StateEmbeddingPayload | None = None,
        **kwargs: object,
    ) -> object:
        generate_method = getattr(self.qwen_model, "generate")
        with self._hook_lock:
            self._clear_captures()
            try:
                with (
                    self._patched_state_embeddings(state_embedding_payload),
                    self._patched_video_features(),
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

    def encode_video_batch_raw(
        self,
        pixel_values_videos: Tensor,
        video_grid_thw: Tensor,
    ) -> RawVideoFeatureBatch:
        """Run HF ViT/Main Merger/DeepStack once without invoking the Fast Adapter."""

        with self._hook_lock:
            self._clear_captures()
            main, deepstack = self.feature_owner.get_video_features(
                pixel_values_videos,
                video_grid_thw,
            )
            return self.video_boundary.capture_raw(main, deepstack, video_grid_thw)

    def prepare_raw_video_batch(
        self,
        raw: RawVideoFeatureBatch,
    ) -> tuple[PreparedVisualChunk, ...]:
        with self._hook_lock:
            return self.video_boundary.prepare_raw_batch(raw)

    def prepare_raw_visual_chunk(self, raw: RawVisualChunk) -> PreparedVisualChunk:
        with self._hook_lock:
            return self.video_boundary.prepare_raw_chunk(raw)

    @contextmanager
    def _patched_video_features(
        self,
    ) -> Iterator[None]:
        with self._hook_lock:
            owner = self.feature_owner
            had_instance_method = "get_video_features" in vars(owner)
            original_instance_method = vars(owner).get("get_video_features")
            original = owner.get_video_features

            def intercepted(
                pixel_values_videos: Tensor,
                video_grid_thw: Tensor | None = None,
            ) -> tuple[Sequence[Tensor], Sequence[Tensor]]:
                main, deepstack = original(pixel_values_videos, video_grid_thw)
                return self.video_boundary.intercept_features(
                    main,
                    deepstack,
                    video_grid_thw,
                )

            owner.get_video_features = intercepted  # type: ignore[method-assign]
            try:
                yield
            finally:
                if had_instance_method:
                    owner.get_video_features = original_instance_method  # type: ignore[method-assign,assignment]
                else:
                    delattr(owner, "get_video_features")

    @contextmanager
    def _patched_state_embeddings(
        self,
        payload: StateEmbeddingPayload | None = None,
    ) -> Iterator[None]:
        if payload is None:
            yield
            return
        with self._hook_lock:
            get_embeddings = getattr(self.qwen_model, "get_input_embeddings", None)
            embedding_layer = get_embeddings()

            def scatter_state(
                _module: nn.Module,
                module_args: tuple[object, ...],
                output: object,
            ) -> object:
                actual_ids = cast(Tensor, module_args[0])
                output = cast(Tensor, output)
                if actual_ids.ndim == 2 and actual_ids.shape[1] == 1:
                    return output
                state_mask = payload.state_position_mask.to(device=output.device)
                expanded_mask = state_mask.unsqueeze(-1).expand_as(output)
                state_values = payload.state_embeddings.to(
                    device=output.device,
                    dtype=output.dtype,
                )
                scattered = output.masked_scatter(expanded_mask, state_values)
                return scattered

            handle = embedding_layer.register_forward_hook(scatter_state)
            try:
                yield
            finally:
                handle.remove()

    def _clear_captures(self) -> None:
        self.video_boundary.last_output = None
        self.video_boundary.last_prepared = None


def _resolve_feature_owner(model: nn.Module) -> QwenFeatureOwner:
    candidate = getattr(model, "model", model)
    if not callable(getattr(candidate, "get_video_features", None)):
        raise TypeError("Qwen model must expose get_video_features on itself or .model")
    return cast(QwenFeatureOwner, candidate)


def _build_merged_metadata(video_grid_thw: Tensor, merge_size: int) -> MergedVideoMetadata:
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


def _left_aligned_mask(token_counts: Sequence[int], width: int, device: torch.device) -> Tensor:
    positions = torch.arange(width, device=device).unsqueeze(0)
    counts = torch.tensor(tuple(token_counts), dtype=torch.int64, device=device).unsqueeze(1)
    return positions < counts
