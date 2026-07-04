from pathlib import Path
import sys
import pytest

# Pytest puts tests directory at sys.path[0], which shadows our project modules.
# Ensure the project root is at the front of sys.path to find the real ebaby/dvdflip packages.
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT in sys.path:
    sys.path.remove(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

# Also restore '' (current dir) at position 1
if '' not in sys.path:
    sys.path.insert(1, '')

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
