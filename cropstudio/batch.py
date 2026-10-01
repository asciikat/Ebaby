"""In-memory buffering of crop shots into DVDs of three (Back, Front, Inside).

Keyed by slot so re-cropping (Prev -> Next) overwrites rather than duplicating.
Pure logic: no disk, no network — the caller writes the returned FlushRequest.
"""
from dataclasses import dataclass


@dataclass
class FlushRequest:
    dvd_index: int
    shots: dict          # {slot:int -> bytes}, 1..3 entries


class BatchState:
    def __init__(self):
        self.current_dvd_index = None
        self.buffer = {}                 # slot -> bytes

    def add(self, dvd_index: int, slot: int, data: bytes):
        """Buffer a shot. Return a FlushRequest if a DVD became ready, else None."""
        flush = None
        if (self.current_dvd_index is not None
                and dvd_index != self.current_dvd_index and self.buffer):
            # A new DVD started; the previous one is done (possibly partial).
            flush = FlushRequest(self.current_dvd_index, dict(self.buffer))
            self.buffer = {}
        self.current_dvd_index = dvd_index
        self.buffer[slot] = data
        if flush is None and len(self.buffer) == 3:
            flush = FlushRequest(self.current_dvd_index, dict(self.buffer))
            self.buffer = {}
            self.current_dvd_index = None
        return flush

    def finish(self):
        """Flush a trailing partial DVD (1-2 shots). None if nothing buffered."""
        if not self.buffer:
            return None
        req = FlushRequest(self.current_dvd_index, dict(self.buffer))
        self.buffer = {}
        self.current_dvd_index = None
        return req
