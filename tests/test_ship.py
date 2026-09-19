import io
import os
import subprocess
import tarfile

import pytest

from mslurm.ship import ShipError, archive, snapshot


def git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    git("init", "-q", cwd=tmp_path)
    git("config", "user.email", "t@t", cwd=tmp_path)
    git("config", "user.name", "t", cwd=tmp_path)
    (tmp_path / ".gitignore").write_text(".venv/\n*.log\n")
    (tmp_path / "train.py").write_text("print(1)\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "eval.py").write_text("print(2)\n")
    git("add", "-A", cwd=tmp_path)
    git("commit", "-qm", "init", cwd=tmp_path)
    return tmp_path


def names(data: bytes) -> set[str]:
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        return {m.name for m in tar.getmembers() if m.isfile()}


def test_clean_tree_matches_head(repo):
    snap = snapshot(str(repo))
    assert not snap.dirty
    assert snap.subdir == ""
    assert snap.head is not None
    head_tree = subprocess.run(["git", "rev-parse", "HEAD^{tree}"], cwd=repo, capture_output=True, text=True).stdout.strip()
    assert snap.tree == head_tree
    assert names(archive(snap)) == {".gitignore", "train.py", "sub/eval.py"}


def test_dirty_and_untracked_included_ignored_excluded(repo):
    (repo / "train.py").write_text("print(3)\n")
    (repo / "new.py").write_text("x\n")
    (repo / ".venv").mkdir()
    (repo / ".venv" / "big").write_text("no\n")
    (repo / "run.log").write_text("no\n")
    snap = snapshot(str(repo / "sub"))
    assert snap.dirty
    assert snap.subdir == "sub"
    assert names(archive(snap)) == {".gitignore", "train.py", "sub/eval.py", "new.py"}
    # the real index is untouched
    status = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True).stdout
    assert "?? new.py" in status
    # same content -> same hash
    assert snapshot(str(repo)).tree == snap.tree


def test_outside_git(tmp_path):
    with pytest.raises(ShipError):
        snapshot(str(tmp_path))


def test_repo_without_commits(tmp_path):
    git("init", "-q", cwd=tmp_path)
    (tmp_path / "a.py").write_text("x\n")
    snap = snapshot(str(tmp_path))
    assert snap.head is None and snap.dirty
    assert names(archive(snap)) == {"a.py"}
