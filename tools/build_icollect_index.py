"""One-time (resumable) crawl of icollecteverything.com's movie index into a
local sqlite barcode index: barcode -> (item_id, title).

The site's robots.txt allows crawling (Disallow: empty). ~23.8k index pages x
50 entries. Run from WSL:
  python3 tools/build_icollect_index.py
Re-running resumes where it left off (done pages are recorded). The output
lands in data/icollect.sqlite, which ebaby.stages.icollect reads at lookup
time — until the file exists the iCollect card line simply doesn't appear.
"""
import re
import sqlite3
import sys
import threading
import time
from pathlib import Path

from curl_cffi import requests as creq

BASE = "https://www.icollecteverything.com"
LAST_PAGE = 23823          # measured 2026-07-12 (binary search of the index)
WORKERS = 4
DB = Path(__file__).resolve().parent.parent / "data" / "icollect.sqlite"

ENTRY = re.compile(
    r'/db/item/movie/(\d+)/">([^<]+)</a>\s*\(Barcode:\s*([0-9A-Za-z]+)\)')

_tls = threading.local()


def _session():
    if getattr(_tls, "sess", None) is None:
        s = creq.Session(impersonate="chrome")
        s.cookies.set("ice_human", "1", domain="www.icollecteverything.com")
        _tls.sess = s
    return _tls.sess


def crawl_page(page):
    for attempt in range(3):
        try:
            r = _session().get(f"{BASE}/db/items/movie/{page}/", timeout=30)
            if r.status_code == 200:
                return ENTRY.findall(r.text)
        except Exception:
            pass
        _tls.sess = None           # fresh session on retry
        time.sleep(2 * (attempt + 1))
    return None                    # page failed — left un-done for a re-run


def main():
    DB.parent.mkdir(parents=True, exist_ok=True)
    # workers share this connection; every use is inside `lock` below
    con = sqlite3.connect(DB, check_same_thread=False)
    con.execute("CREATE TABLE IF NOT EXISTS items"
                " (barcode TEXT, item_id INTEGER, title TEXT,"
                "  PRIMARY KEY (barcode, item_id))")
    con.execute("CREATE TABLE IF NOT EXISTS pages_done (page INTEGER PRIMARY KEY)")
    done = {p for (p,) in con.execute("SELECT page FROM pages_done")}
    todo = [p for p in range(LAST_PAGE + 1) if p not in done]
    print(f"{len(done)} pages done, {len(todo)} to go", flush=True)

    lock = threading.Lock()
    counter = {"pages": 0, "rows": 0}

    def worker(pages):
        for page in pages:
            entries = crawl_page(page)
            time.sleep(0.25)       # polite: ~16 req/s across 4 workers max
            if entries is None:
                continue
            with lock:
                con.executemany(
                    "INSERT OR IGNORE INTO items VALUES (?, ?, ?)",
                    [(bc, int(iid), title.strip()) for iid, title, bc in entries])
                con.execute("INSERT OR IGNORE INTO pages_done VALUES (?)", (page,))
                counter["pages"] += 1
                counter["rows"] += len(entries)
                if counter["pages"] % 200 == 0:
                    con.commit()
                    print(f"{counter['pages']}/{len(todo)} pages, "
                          f"{counter['rows']} entries", flush=True)

    threads = [threading.Thread(target=worker, args=(todo[i::WORKERS],))
               for i in range(WORKERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    con.commit()
    total = con.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    print(f"DONE: {total} barcode entries in {DB}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
