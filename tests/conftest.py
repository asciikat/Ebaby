from pathlib import Path
import pytest

SAMPLES = Path(__file__).resolve().parent.parent / "samples"

@pytest.fixture
def sample_dngs():
    files = sorted(SAMPLES.glob("*.dng"))
    if len(files) < 3:
        pytest.skip("sample DNGs not present")
    return files

@pytest.fixture
def sample_front(sample_dngs):
    return next(p for p in sample_dngs if p.name.endswith("153112.dng"))

@pytest.fixture
def sample_back(sample_dngs):
    return next(p for p in sample_dngs if p.name.endswith("153120.dng"))

@pytest.fixture
def sample_center(sample_dngs):
    return next(p for p in sample_dngs if p.name.endswith("153137.dng"))
