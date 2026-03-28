"""Tests for cross-layer operators (src/alns/crosslayer.py)."""
import numpy as np
import pytest

from src.alns.crosslayer import cross_dispatch
from src.data.constants import N_CROSS_OPS


def test_cross_ops_count():
    assert N_CROSS_OPS == 5


def test_cross_dispatch_is_callable():
    assert callable(cross_dispatch)
