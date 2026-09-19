import pytest

from mslurm import MslurmError
from mslurm.slurm import (
    merge_jobs,
    parse_duration,
    parse_gpus,
    parse_gres_types,
    parse_sacct,
    parse_sinfo,
    parse_squeue,
    sacct_since,
    slurm_time,
)

SQUEUE = """\
154980|train|RUNNING|main|0:11:00|4:00:00|1|node07|None|/home/u/mslurm/jobs/x|gres/gpu:rtx3090:1|2026-09-17T10:00:00
154981|sweep|PENDING|main|0:00|4:00:00|1||(Priority)|/home/u/mslurm/jobs/y|gres/gpu:2|2026-09-17T10:01:00
154982_[0-3]|arr|PENDING|main|0:00|1:00:00|1||Resources|/home/u/mslurm/jobs/z|N/A|2026-09-17T10:02:00
"""

SACCT = """\
154914|eval-sweep|COMPLETED|main|00:11:00|03:00:00|1|node03||/home/u/.mslurm_scratch/5bc4|billing=8,cpu=8,gres/gpu:rtx3090=1,gres/gpu=1,mem=32G,node=1|2026-09-15T10:23:00|2026-09-15T10:34:46
154974|eval-sweep|CANCELLED by 1000|main|00:00:00|03:00:00|1|None assigned|None|/home/u/x|billing=1,cpu=1,node=1|2026-09-15T13:38:00|2026-09-15T13:38:37
154980|train|RUNNING|main|00:10:00|04:00:00|1|node07|None|/home/u/mslurm/jobs/x|cpu=4,gres/gpu:rtx3090=1,node=1|2026-09-17T10:00:00|Unknown
"""

SINFO = """\
interactive|up|1-00:00:00|idle|gpu:rtx3080:1
main*|up|3-00:00:00|mix|gpu:rtx3090:8(S:0-1)
main*|up|3-00:00:00|idle|gpu:rtx3090:8(S:0-1)
main*|up|3-00:00:00|alloc|gpu:rtx6000Ada:4(S:0),gpu:a6000:4(S:1)
main*|up|3-00:00:00|idle~|gpu:a5000:4(S:0)
main*|up|3-00:00:00|down*|(null)
"""


def test_parse_gpus():
    assert parse_gpus("gres/gpu:rtx3090:1") == "rtx3090:1"
    assert parse_gpus("gres/gpu:2") == "gpu:2"
    assert parse_gpus("N/A") == ""
    assert parse_gpus("billing=8,cpu=8,gres/gpu:rtx3090=1,gres/gpu=1,mem=32G,node=1") == "rtx3090:1"
    assert parse_gpus("cpu=8,gres/gpu=2,node=1") == "gpu:2"
    assert parse_gpus("cpu=1,node=1") == ""


def test_parse_gres_types():
    assert parse_gres_types("gpu:rtx6000Ada:4(S:0),gpu:a6000:4(S:1)") == {"rtx6000Ada": 4, "a6000": 4}
    assert parse_gres_types("gpu:8") == {"gpu": 8}
    assert parse_gres_types("(null)") == {}


def test_parse_squeue():
    jobs = parse_squeue(SQUEUE)
    assert [j.id for j in jobs] == ["154980", "154981", "154982_[0-3]"]
    assert jobs[0].gpus == "rtx3090:1"
    assert jobs[0].where == "node07"
    assert jobs[1].where == "(Priority)"
    assert jobs[2].gpus == ""


def test_parse_sacct_and_merge():
    history = parse_sacct(SACCT)
    assert history[1].state == "CANCELLED"
    assert history[0].gpus == "rtx3090:1"
    merged = merge_jobs(parse_squeue(SQUEUE), history)
    ids = [j.id for j in merged]
    assert ids == ["154980", "154982_[0-3]", "154981", "154974", "154914"]
    running = next(j for j in merged if j.id == "154980")
    assert running.elapsed == "0:11:00"  # squeue row wins


def test_parse_sinfo():
    parts = {p.name: p for p in parse_sinfo(SINFO)}
    assert parts["main"].default and not parts["interactive"].default
    main = parts["main"]
    assert main.nodes == 5 and main.idle == 2
    assert main.gpus == {"rtx3090": 16, "rtx6000Ada": 4, "a6000": 4, "a5000": 4}
    assert main.gpus_idle == {"rtx3090": 8, "a5000": 4}


def test_durations():
    assert parse_duration("30m") == 1800
    assert parse_duration("1.5h") == 5400
    assert parse_duration("3d") == 3 * 86400
    with pytest.raises(MslurmError):
        parse_duration("soon")
    assert slurm_time("5m") == "0:05:00"
    assert slurm_time("2h") == "2:00:00"
    assert slurm_time("1d") == "1-00:00:00"
    assert slurm_time("36h") == "1-12:00:00"
    assert slurm_time("01:30:00") == "01:30:00"
    assert slurm_time("2-00:00") == "2-00:00"
    assert sacct_since("1d") == "now-1days"
    assert sacct_since("6h") == "now-6hours"
    assert sacct_since("90m") == "now-90minutes"
