"""Shared fixtures for the ebaby test suite.

The real-RAW fixtures point at untracked local DNGs under samples/ and SKIP
when they aren't present, so the suite stays green on a fresh checkout while
still exercising the true RAW path on machines that have the photos.
"""
from pathlib import Path

import pytest

SAMPLES = Path(__file__).resolve().parent.parent.parent / "samples"


@pytest.fixture
def sample_front():
    files = sorted(SAMPLES.glob("*.dng"))
    if not files:
        pytest.skip("sample DNGs not present")
    return files[0]
