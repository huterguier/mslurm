# mslurm

One command to submit to and inspect several Slurm clusters from a laptop or
workstation. Everything happens over ssh with a reused connection; there is no
daemon and nothing to install on the clusters.

## Install

```sh
uv tool install git+https://github.com/huterguier/mslurm
mslurm init               # writes ~/.config/mslurm/clusters.yaml, then edit it
mslurm info               # should list partitions and GPUs
```

Needs `ssh`, `rsync` and `git` locally, and key-based ssh access to each
cluster. The remote login shell must be bash, zsh or fish; csh and tcsh are
not supported. Works on Linux and macOS.

## Configure

One entry per cluster in `~/.config/mslurm/clusters.yaml`. Only `host` is
required; `mslurm init` writes a commented example with every key.

```yaml
default: alpha
clusters:
  alpha:
    host: alpha               # ssh host or ~/.ssh/config alias
    scratch: ~/mslurm         # where code snapshots and job dirs go on the cluster
    account: my-account       # put into every #SBATCH header
    partition: main
    defaults:                 # used when the flag is not given
      time: "4:00:00"
      cpus-per-task: 4
    gpus:                     # short names for -G
      "3090": rtx3090
    setup: |                  # runs in the job dir before your command
      uv sync
      export PATH="$(dirname "$(uv python find)"):$PATH"
```

## Use

```sh
mslurm submit -G 3090 -t 2h -- python train.py --lr 1e-3     # from inside a git repo
mslurm submit -M alpha -a 0-9 -- python sweep.py               # array job on a named cluster
mslurm submit job.sh -- arg1                                  # your own sbatch script
mslurm queue                    # running, pending and recently finished jobs on all clusters
mslurm logs alpha:154980 -f       # follow the output
mslurm show alpha:154980
mslurm pull alpha:154980          # outputs -> ./runs/alpha/154980/, never into the source tree
mslurm pull alpha:154980 .        # outputs only, merged into the cwd (no logs or batch script)
mslurm cancel alpha:154980
mslurm clean                    # old job dirs and unused code snapshots (asks first)
mslurm ssh alpha                  # a shell, useful once for hosts that need 2FA
```

Jobs are named `cluster:jobid`. A bare id works when it is in the local index or
only one cluster is configured.

Short flags mean what they mean for `sbatch`: `-M` cluster, `-J` job name,
`-t` time, `-p` partition, `-G` GPUs, `-c` CPUs per task, `-n` tasks, `-N` nodes,
`-a` array, `-A` account, `-q` QOS. Two accept shorthands sbatch does not:
`-t 2h` and `-G 3090:2` (an alias from `clusters.yaml`, expanded to a gres type).
Anything else goes through `--sbatch`, repeatable, e.g. `--sbatch=--exclusive`,
or through a script's own `#SBATCH` lines.

## How a submit works

1. The git working tree (including uncommitted changes, excluding ignored files)
   is snapshotted as a git tree object. Its hash names the snapshot.
2. The snapshot is uploaded to `<scratch>/code/<hash>` unless it is already there.
3. A fresh job directory `<scratch>/jobs/<stamp>` gets a copy of the snapshot,
   the generated `mslurm.sh` and a `mslurm.json` with the metadata.
4. `sbatch` runs in the directory matching where you ran `mslurm submit`, so
   relative paths behave as they do locally. Output goes to `slurm-<jobid>.out`.

Options fill the `#SBATCH` header on top of the cluster's `defaults:`.
`mslurm submit --dry-run` prints the script without submitting.

## Develop

```sh
uv sync
uv run pytest
```

## License

MIT
