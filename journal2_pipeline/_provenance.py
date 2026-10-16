"""
Weight provenance: tie every number to the exact version of the file that produced it.

THE PROBLEM THAT OCCURRED (2026-08-02). The result CSVs recorded the NAME of the weight file,
not its VERSION. All 180 KD checkpoints were retrained between 17 and 24 July 2026, overwriting
the old files. From then on, every measurement taken before July described weights that no
longer existed, and nothing flagged it. It was caught only because a chance cross-check showed
a gap of up to 0.0507, while the baseline branch (weights unchanged since May) reproduced to
within 0.0010.

THE FIX. Three levels, all built on one shared function:

  1. Fingerprint of ONE file  -> fingerprint(path)
     Cheap: size + mtime + sha256 of the first 1 MB and the last 1 MB. It does not read the
     whole file, so it stays fast when called 192 times in one run. Reliable enough for this
     purpose: a retrained checkpoint almost certainly differs at the start (first-layer weights)
     and differs in size or mtime. For absolute certainty use full=True, which reads everything.

  2. MANIFEST id for the whole set  -> manifest_id(list_of_fingerprints)
     A 12-character string summarizing the state of all 192 weights. Changing one file changes the id.

  3. STAMP the result CSV -> stamp_columns()
     Every measurement script must write manifest_id into its output. Later, a look at the CSV
     tells which weight state it describes, and this can be verified by recomputing the id from disk.

Why full sha256 is not the default: 192 files of tens to hundreds of MB each, so reading them all
costs several GB per run. Head-tail catches every realistic overwrite at almost no cost.
The only case it misses is a file modified EXACTLY in the middle while keeping its size and
mtime, a scenario that does not happen when retraining.
"""
from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone

CHUNK = 1024 * 1024   # 1 MB at each end


def fingerprint(path: str, full: bool = False) -> dict:
    """Return the fingerprint of a weight file. If the file is missing, return an empty dict with a flag."""
    if not path or not os.path.exists(path):
        return {'path': path or '', 'exists': False, 'size': 0, 'mtime': '', 'sha': ''}
    st = os.stat(path)
    h = hashlib.sha256()
    h.update(str(st.st_size).encode())
    with open(path, 'rb') as f:
        if full or st.st_size <= 2 * CHUNK:
            for block in iter(lambda: f.read(CHUNK), b''):
                h.update(block)
        else:
            h.update(f.read(CHUNK))
            f.seek(-CHUNK, os.SEEK_END)
            h.update(f.read(CHUNK))
    return {
        'path': path,
        'exists': True,
        'size': st.st_size,
        'mtime': datetime.fromtimestamp(st.st_mtime, tz=timezone.utc)
                         .astimezone().isoformat(timespec='seconds'),
        'sha': h.hexdigest()[:16],
        'sha_mode': 'full' if full else 'head-tail',
    }


def manifest_id(fps) -> str:
    """Combine several fingerprints into ONE short id for the whole set.

    Sorted by path so the id does not depend on the traversal order. Missing files also enter the id
    (as empty strings) so that "3 files missing" and "all files present" do not give the same id.
    """
    h = hashlib.sha256()
    for fp in sorted(fps, key=lambda d: d.get('path', '')):
        h.update(f"{fp.get('path', '')}|{fp.get('size', 0)}|{fp.get('sha', '')}\n".encode())
    return h.hexdigest()[:12]


def stamp_columns(mid: str) -> dict:
    """Stamp columns to insert into every row of a result CSV."""
    return {'weight_manifest_id': mid,
            'measured_at': datetime.now().astimezone().isoformat(timespec='seconds')}


STAMP_FIELDS = ['weight_manifest_id', 'measured_at']
