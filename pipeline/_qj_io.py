"""
where-teacher-acts: the only way this repository writes a checkpoint.

Three rules, enforced where the file is written rather than trusted to the caller:

  1. Checkpoints go under weights/qj and nowhere else. A path whose real location lies inside a
     read-only input (weights/phase3, weights/phase3_kd, weights/phase2, which are symlinks into
     the sibling repository of an earlier study) is refused however it is spelled. On 2026-08-14 a
     sweep that wrote into weights/phase3_kd under the canonical names destroyed four checkpoints
     of an earlier study; this module exists so that cannot happen from here.
  2. One run, one directory, claimed exclusively. A directory that already holds a finished run
     is refused unless the caller asks to overwrite it; a directory another process is using
     (lock file present) is always refused, unless the lock was written on this host by a process
     that no longer exists (a run killed hard), in which case the stale lock is removed and said so;
     the check and the takeover are serialized (flock), so two claimers cannot both take it over.
  3. Writes are atomic: a crash mid-save leaves the previous file or no file, never half a file.
  4. A finished run holds only what this attempt wrote. Claiming a directory first marks it running,
     then deletes the checkpoint files a failed or interrupted earlier attempt left there (model.pth
     and its sidecars), and finish() refuses unless this attempt saved model.pth itself, so a run can
     never be marked done on a stale checkpoint. run.json records the scheduler's attempt token
     (environment QJ_ATTEMPT), so the scheduler reads only its own attempt's outcome.

run.json in each run directory records what the run was, where it started from (with the start
checkpoint's fingerprint), the repository commit, and its status: running, done or failed.
"""
import os
import json
import time
import fcntl
import socket
import contextlib
import subprocess

WRITE_ROOT = os.path.join("weights", "qj")
READ_ONLY_INPUTS = ("weights/phase3", "weights/phase3_kd", "weights/phase2")
LOCK_NAME = ".lock"
META_NAME = "run.json"
PAYLOAD_PREFIX = "model.pth"        # model.pth, model.pth.clip.pt and their .tmp-<pid> leftovers
ATTEMPT_ENV = "QJ_ATTEMPT"          # set by the Part II scheduler for each attempt it launches
REQUIRED = ("model.pth",)           # what finish() demands this attempt saved


def _real(path):
    return os.path.realpath(os.path.abspath(path))


def _inside(path, root):
    p, r = _real(path), _real(root)
    return p == r or p.startswith(r + os.sep)


def assert_writable(path):
    """Raise PermissionError unless `path` resolves under WRITE_ROOT and outside every input."""
    for ro in READ_ONLY_INPUTS:
        if os.path.lexists(ro) and _inside(path, ro):
            raise PermissionError(f"refusing to write {path}: it resolves inside the read-only "
                                  f"input {ro} ({_real(ro)})")
    if not _inside(path, WRITE_ROOT):
        raise PermissionError(f"refusing to write {path}: checkpoints may only be written under "
                              f"{WRITE_ROOT}")


def atomic_save(obj, path):
    """torch.save to a temporary name in the same directory, then rename over the target."""
    import torch
    assert_writable(path)
    tmp = f"{path}.tmp-{os.getpid()}"
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def _stale_lock(path):
    """True only when the lock names this host and a pid that no longer exists."""
    try:
        with open(path) as f:
            host, _, pid = f.read().split()[:3]
        pid = int(pid)
    except (OSError, ValueError):
        return False
    if host != socket.gethostname():
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def take_lock(path, refusal):
    """Create the lock file `path` exclusively and return its fd. If it exists and was left by a dead
    process on this host, take it over; the check and the takeover run under an flock on a sibling
    file, so two processes that find the same dead lock cannot both take it over. Otherwise raise
    FileExistsError(refusal)."""
    try:
        return os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        pass
    with file_lock(path + ".takeover"):
        if not _stale_lock(path):
            raise FileExistsError(refusal)
        print(f"[qj_io] {path}: its process is gone, removing the stale lock", flush=True)
        os.remove(path)
        return os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)


def _repo_head():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


class RunClaim:
    """Exclusive ownership of one run directory for the lifetime of a `with` block.

        with RunClaim(run_dir(...), meta={...}) as claim:
            ... training ...
            claim.save(state_dict, "model.pth")
            claim.finish(mAP_sim=...)
    """

    def __init__(self, rdir, meta=None, overwrite=False):
        self.rdir = rdir
        self.meta = dict(meta or {})
        self.overwrite = overwrite
        self._lock = os.path.join(rdir, LOCK_NAME)
        self._meta_path = os.path.join(rdir, META_NAME)
        self._owned = False
        self.saved = set()

    def __enter__(self):
        assert_writable(self.rdir)
        os.makedirs(self.rdir, exist_ok=True)
        if os.path.exists(self._meta_path) and not self.overwrite:
            with open(self._meta_path) as f:
                prev = json.load(f)
            if prev.get("status") == "done":
                raise FileExistsError(f"{self.rdir} already holds a finished run; pass "
                                      f"overwrite=True only to redo it on purpose")
        fd = take_lock(self._lock, f"{self.rdir} is locked by a live process (or one on another host); "
                                   f"check, then delete {self._lock} by hand")
        os.write(fd, f"{socket.gethostname()} pid {os.getpid()} {time.strftime('%Y-%m-%dT%H:%M:%S')}\n".encode())
        os.close(fd)
        self._owned = True
        # mark the directory running BEFORE deleting anything, so an interruption in between can never
        # leave a "done" run.json without its checkpoint
        self._write_meta(status="running", started=time.strftime("%Y-%m-%dT%H:%M:%S"),
                         host=socket.gethostname(), repo_commit=_repo_head(),
                         attempt=os.environ.get(ATTEMPT_ENV, ""))
        stale = sorted(n for n in os.listdir(self.rdir) if n.startswith(PAYLOAD_PREFIX)
                       and os.path.isfile(os.path.join(self.rdir, n)))
        for n in stale:
            os.remove(os.path.join(self.rdir, n))
        if stale:
            print(f"[qj_io] {self.rdir}: removed files of an earlier attempt: {stale}", flush=True)
        return self

    def save(self, obj, name):
        if not self._owned:
            raise RuntimeError("save() outside an active RunClaim")
        if os.sep in name or name.startswith("."):
            raise ValueError(f"file name {name!r} must be a plain name inside the run directory")
        out = atomic_save(obj, os.path.join(self.rdir, name))
        self.saved.add(name)
        return out

    def finish(self, **results):
        missing = [n for n in REQUIRED if n not in self.saved]
        if missing:
            raise RuntimeError(f"finish() refused: this attempt never saved {missing}")
        self._write_meta(status="done", finished=time.strftime("%Y-%m-%dT%H:%M:%S"), **results)

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is not None:
                self._write_meta(status="failed", error=f"{exc_type.__name__}: {exc}")
            elif self.meta.get("status") != "done":
                self._write_meta(status="failed", error="the run ended without finish()")
        finally:
            if self._owned:
                with contextlib.suppress(FileNotFoundError):
                    os.remove(self._lock)
                self._owned = False
        return False

    def _write_meta(self, **fields):
        self.meta.update(fields)
        tmp = f"{self._meta_path}.tmp-{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(self.meta, f, indent=2, sort_keys=True, default=str)
        os.replace(tmp, self._meta_path)


@contextlib.contextmanager
def file_lock(path):
    """Block until this process holds an exclusive lock on `path` (created if missing)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
