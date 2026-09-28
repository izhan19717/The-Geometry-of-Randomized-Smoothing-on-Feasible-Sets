"""Pytest collection rules for the anonymous supplementary archive.

One source-binding test validates the original machine-path-bearing protocol
bytes. Those bytes are intentionally replaced by an anonymous,
project-relative derivative in this archive. The file remains available
because the retained trajectory audit checks its original source hash, but its
config-byte assertions do not apply to the anonymous derivative. The remaining
tests in that module still run.
"""

from __future__ import annotations

import pytest


_SOURCE_BINDING_TEST = (
    "tests/test_iclr2027_safety_gym_all_controller_integration.py::"
    "test_frozen_protocol_binds_sources_and_controller_artifacts"
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip the one assertion that targets the non-anonymous config bytes."""

    for item in items:
        if item.nodeid == _SOURCE_BINDING_TEST:
            item.add_marker(
                pytest.mark.skip(
                    reason="original config paths were normalized for review"
                )
            )
