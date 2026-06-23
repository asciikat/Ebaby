from cropstudio.batch import BatchState


def test_full_dvd_flushes_on_third_slot():
    s = BatchState()
    assert s.add(0, 0, b"back") is None
    assert s.add(0, 1, b"front") is None
    req = s.add(0, 2, b"inside")
    assert req is not None
    assert req.dvd_index == 0
    assert req.shots == {0: b"back", 1: b"front", 2: b"inside"}


def test_resending_same_slot_overwrites():
    s = BatchState()
    s.add(0, 0, b"first")
    s.add(0, 0, b"second")           # Prev -> re-crop -> Next
    s.add(0, 1, b"front")
    req = s.add(0, 2, b"inside")
    assert req.shots[0] == b"second"


def test_higher_dvd_index_flushes_partial_previous():
    s = BatchState()
    s.add(0, 0, b"back")             # only 1 shot for DVD 0
    req = s.add(1, 0, b"next-back")  # DVD 1 begins -> DVD 0 flushes partial
    assert req is not None
    assert req.dvd_index == 0
    assert req.shots == {0: b"back"}


def test_finish_flushes_trailing_partial():
    s = BatchState()
    s.add(0, 0, b"back")
    s.add(0, 1, b"front")
    req = s.finish()
    assert req.shots == {0: b"back", 1: b"front"}
    assert s.finish() is None        # nothing left


def test_finish_empty_returns_none():
    assert BatchState().finish() is None
