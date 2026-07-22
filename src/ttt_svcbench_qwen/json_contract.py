"""Strict JSON value readers shared by annotation and manifest schemas."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast


def object_value(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object with string keys")
    return cast(dict[str, object], value)


def mapping_value(row: Mapping[str, object], key: str) -> dict[str, object]:
    return object_value(row.get(key), key)


def string_value(row: Mapping[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def integer_value(row: Mapping[str, object], key: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def number_value(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be numeric")
    return float(value)


def float_value(row: Mapping[str, object], key: str) -> float:
    return number_value(row.get(key), key)
