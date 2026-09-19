import pytest

from mslurm import MslurmError
from mslurm.config import Cluster, Gpu
from mslurm.render import (
    Resources,
    guess_name,
    render_script,
    resolve_gpus,
    sbatch_args,
    sbatch_directives,
    sbatch_options,
)

CLUSTER = Cluster(
    name="alpha",
    host="alpha",
    account="my-account",
    qos="normal",
    partition="main",
    gpus={"3090": Gpu("rtx3090"), "small": Gpu("rtx3080", partition="interactive")},
    setup='export PATH="$HOME/.local/bin:$PATH"\nuv sync',
    defaults={"time": "4:00:00", "cpus-per-task": "4", "mem": "16G"},
    env={"WANDB_MODE": "online"},
)


def test_resolve_gpus():
    assert resolve_gpus(CLUSTER, None) == (None, None)
    assert resolve_gpus(CLUSTER, "2") == ("gpu:2", None)
    assert resolve_gpus(CLUSTER, "3090") == ("gpu:rtx3090:1", None)
    assert resolve_gpus(CLUSTER, "3090:4") == ("gpu:rtx3090:4", None)
    assert resolve_gpus(CLUSTER, "a5000:2") == ("gpu:a5000:2", None)
    assert resolve_gpus(CLUSTER, "small") == ("gpu:rtx3080:1", "interactive")
    with pytest.raises(MslurmError):
        resolve_gpus(CLUSTER, "3090:x")


def test_options_merge_defaults_and_flags():
    opts = dict(sbatch_options(CLUSTER, Resources(name="train", time="2h", gpus="3090", mem="64G")))
    assert opts["job-name"] == "train"
    assert opts["account"] == "my-account"
    assert opts["qos"] == "normal"
    assert opts["partition"] == "main"
    assert opts["time"] == "2:00:00"
    assert opts["cpus-per-task"] == "4"  # default
    assert opts["mem"] == "64G"  # flag wins
    assert opts["gres"] == "gpu:rtx3090:1"
    assert opts["output"] == "slurm-%j.out"


def test_alias_partition_and_array_output():
    opts = dict(sbatch_options(CLUSTER, Resources(name="x", gpus="small", array="0-9")))
    assert opts["partition"] == "interactive"
    assert opts["output"] == "slurm-%A_%a.out"
    opts = dict(sbatch_options(CLUSTER, Resources(name="x", gpus="small", partition="main")))
    assert opts["partition"] == "main"


def test_script_mode_leaves_defaults_out():
    opts = sbatch_options(CLUSTER, Resources(time="1h"), comment="mslurm:abc", script_mode=True)
    keys = [k for k, _ in opts]
    assert keys == ["account", "qos", "time", "comment"]
    assert sbatch_args(opts) == ["--account=my-account", "--qos=normal", "--time=1:00:00", "--comment=mslurm:abc"]


def test_script_directives_win_over_profile():
    script = "#!/bin/bash\n#SBATCH -A other-acc\n#SBATCH --qos=high  # comment\n#SBATCH --partition gpu\necho hi\n"
    has = sbatch_directives(script)
    assert has == {"account", "qos", "partition"}
    opts = dict(sbatch_options(CLUSTER, Resources(), script_mode=True, script_has=has))
    assert "account" not in opts and "qos" not in opts and "partition" not in opts
    # an explicit flag still overrides the script, like sbatch's command line does
    opts = dict(sbatch_options(CLUSTER, Resources(qos="low"), script_mode=True, script_has=has))
    assert opts["qos"] == "low"
    # without the directives the profile fills them in
    opts = dict(sbatch_options(CLUSTER, Resources(), script_mode=True, script_has=set()))
    assert opts["account"] == "my-account" and opts["qos"] == "normal"


def test_extra_flags():
    opts = sbatch_options(CLUSTER, Resources(name="x", extra=["--exclusive", "--constraint=a100"]))
    assert ("exclusive", None) in opts
    assert ("constraint", "a100") in opts
    assert "--exclusive" in sbatch_args(opts)


def test_render_script():
    text = render_script(
        CLUSTER,
        Resources(name="train", gpus="2", env={"LR": "1e-3"}),
        ["python", "train.py", "--tag", "a b"],
        code="/home/u/mslurm/code/abc",
        comment="mslurm:abc",
    )
    lines = text.splitlines()
    assert lines[0] == "#!/bin/bash"
    assert "#SBATCH --job-name=train" in lines
    assert "#SBATCH --gres=gpu:2" in lines
    assert "#SBATCH --comment=mslurm:abc" in lines
    assert "export WANDB_MODE=online" in lines
    assert "export LR=1e-3" in lines
    assert "uv sync" in lines
    assert lines[-1] == "python train.py --tag 'a b'"
    assert lines.index("uv sync") < lines.index(lines[-1])


def test_guess_name():
    assert guess_name(["python", "train.py", "--lr", "1e-3"], "repo") == "train"
    assert guess_name(["uv", "run", "scripts/eval_dot.py"], "repo") == "eval_dot"
    assert guess_name(["python", "-m", "proj.train"], "repo") == "train"
    assert guess_name(["CUDA_VISIBLE_DEVICES=0", "python", "x.py"], "repo") == "x"
    assert guess_name(["--weird"], "my repo") == "my-repo"
