---
License: Copyright © 2026 Durham University, SHAREing Project, MIT Licensed
Creator: Fawada Qaiser
Contributors: Fawada Qaiser
Summary: Pre-assessment report for astro-cluster-detector (in progress)
Template: SHAREing-DRI/assessment-template, pre-assessment-report.md
---

# Pre-assessment: astro-cluster-detector

> **Status: in progress (2026-09-27).** Filled from a read-through of the code owner's `scripts/`
> directory and correspondence with the owner. No benchmark has been run yet. Items marked
> **TBC** are awaiting the owner or a COSMA8 measurement.

## Assessment objective

An internal review of `astro-cluster-detector` (version: **TBC — no tag or commit reference yet**),
a U-Net heatmap detector for galaxy clusters in weak-lensing convergence maps (DUSTGRAIN N-body
simulations), using a training benchmark to be defined from `scripts/train.py`. Performed as part of
the SHAREing project (UKRI Digital Research Infrastructure) by Fawada Qaiser, DiRAC / Durham University.
Assessment started September 2026.

## Disclaimers

1. This report is not a commentary on code quality, but an indicator of the quality of the current
   SHAREing testing methodology as of September 2026.
2. The pre-assessment is only a preliminary assessment of submission suitability and does not
   guarantee a full assessment.
3. The code is at an early stage of development and has so far been run only on the owner's laptop.

## Table of contents

- [ ] [1: Benchmark setup](#1-benchmark-setup)
- [ ] [2: Description of working environment](#2-description-of-working-environment)
- [ ] [3: Compiler setup and optimisations](#3-compiler-setup-and-optimisations)
- [ ] [4: Computational complexity and scaling](#4-computational-complexity-and-scaling)
- [ ] [5: Memory, storage and I/O](#5-memory-storage-and-io)
- [ ] [6: Additional comments from submitter](#6-additional-comments-from-submitter)
- [ ] [7: Pre-assessment outcome](#7-pre-assessment-outcome)

## Submission status

| Item requested | Received | Notes |
|---|---|---|
| Code repository | Zip archive | Completeness being verified: all scripts import the `astrocluster` package from `src/`, and read `configs/default.yaml` |
| Frozen version (tag/commit) | No | Requested |
| Data subset | 10 simulations (~160 MB) | Format `Data/*.npz` + `truth/*.txt`; full dataset ~150 GB |
| Environment specification | No | None exists yet; `pip freeze` / `conda env export` from the owner's working environment requested |
| Typical run: command, config, epochs, runtime, hardware | Partial | Run on owner's laptop with a GPU; GPU model and CUDA version TBC |
| Representativeness of benchmark | No | Requested: intended production scale (simulations, epochs) |
| How cost scales with problem size | No | Requested |
| Switch to disable per-epoch outputs | No | None visible in the scripts |

## 1: Benchmark setup

### Fetch and build program

Python code; there is no build step. Expected layout (inferred from the scripts, which add
`<repo>/src` to `sys.path`):

```text
<repo>/
  scripts/   evaluate.py, inspect_dataset.py, make_submission.py, predict.py,
             run_256_workflow.py, train.py, visualize_batch.py, visualize_sample.py
  src/astrocluster/   checkpoint, data, losses, metrics, models, postprocess,
                      targets, utils, visualization
  configs/default.yaml
  notebooks/end_to_end_workflow.ipynb
```

Dependencies visible from imports: `torch`, `numpy`, `pyyaml`, `tqdm`, `matplotlib`. Versions TBC;
further dependencies of `astrocluster` TBC.

### Fetch and run benchmark

The owner's reference workflow is `scripts/run_256_workflow.py`. It takes the first 256 complete
simulations and runs six steps in sequence: dataset inspection, visualisation, training (200 train /
50 validation), evaluation with a threshold/radius sweep, prediction on 6 held-out maps, and a
threshold sweep. It stops if fewer than 256 simulations are present, so it cannot run on the
10-simulation subset.

The guidebook asks for a single well-defined benchmark. **Proposed benchmark: training only.**

```bash
python scripts/train.py --config <benchmark_config>.yaml --root <data_root> --out <out_dir>
```

with a config derived from `configs/default.yaml` that splits the 10 simulations (e.g. 8 train /
2 validation) and caps run time via `training.epochs` and `training.max_train_batches`.
Inference (`scripts/evaluate.py`) is a candidate second benchmark if the owner considers it important.
Both to be agreed with the owner.

Console output per epoch (from `train.py`): tqdm progress bars for training and validation, then one
summary line of the form

```text
Epoch <e>/<E>
  train_loss=... val_loss=... F1=... P=... R=... pred/map=... lr=...
```

### Reference architecture

The owner runs the code on a laptop with a GPU (model TBC; whether it is an NVIDIA/CUDA device is TBC).
There are no reference runtimes on the target system. The target system for this assessment is
COSMA8 (NVIDIA A100 GPU nodes). Because configuration values were chosen on a laptop GPU, batch size
and similar settings may under-use an A100; the owner's configuration will be benchmarked as supplied,
and the owner asked for a configuration appropriate to a larger GPU.

## 2: Description of working environment

### Hardware information

COSMA8 GPU nodes, DiRAC Memory Intensive service, Durham University. Details to be collected on a
compute node (not the login node) with `scripts/probe_node.sh`.

| Specification | Per node |
|---|---|
| Processors | TBC |
| Hardware accelerators | TBC × NVIDIA A100 (40 or 80 GB variant TBC) |
| Clock speed per CPU | TBC |
| Sockets | TBC |
| Cores | TBC |
| RAM | TBC |
| Local storage | TBC |

NUMA layout, cache sizes and GPU topology: TBC from probe output.

### Libraries and modules

To be recorded from `module list` and `pip freeze` on COSMA8. The owner's environment is a laptop
install, so the COSMA8 environment will be built by the assessor: the CUDA build of the owner's
PyTorch version, with other packages matched to the owner's versions where possible. All differences
will be recorded here.

### Assessment tools

1. Pre-assessment:
   - `/usr/bin/time -v` (wall time, peak resident memory)
   - `nvidia-smi` (GPU memory, utilisation)
2. High-level assessment:
   - Nsight Compute (`ncu`) for GPU FLOPs, restricted to a few iterations via NVTX ranges; see the
     methodology note in the README
   - Likwid (`likwid-bench`, `likwid-perfctr`) for host CPU peak and achieved FLOPs
   - Darshan for I/O time (applicability to a non-MPI Python process TBC)
   - Wall-clock timing inside the run with `torch.cuda.synchronize()`, excluding start-up
3. Low-level assessment:
   - Nsight Systems (`nsys`) for CPU/GPU timeline and data-loading stalls
   - PyTorch profiler

Open question: whether normal users on COSMA8 can read GPU performance counters
(`ERR_NVGPUCTRPERM`); checked by `probe_node.sh`.

## 3: Compiler setup and optimisations

Not a compiled code. The equivalent settings that affect performance, as visible in the scripts:

| Setting | Value in scripts | Note |
|---|---|---|
| Automatic mixed precision (AMP) | On by default when device is CUDA | Uses the older `torch.cuda.amp` API; FP16 arithmetic on tensor cores |
| Determinism | `set_seed(..., deterministic=True)` by default in every script | Likely disables `cudnn.benchmark` and forces deterministic kernels; to be confirmed in `astrocluster/utils.py` |
| TF32 | Not set in scripts | PyTorch defaults apply (TF32 allowed for cuDNN convolutions) |
| Pinned memory, non-blocking transfers | On when device is CUDA | |
| DataLoader workers | `num_workers` default 2 | |
| Tiled inference | Optional (`inference.tiled`) | Inference only |

Debug symbols (`-g`) do not apply; NVTX ranges will be added in the benchmark harness for profiling.

## 4: Computational complexity and scaling

Parallel model: a single Python process on a single GPU, with DataLoader worker processes for data
loading. No MPI, DDP or multi-GPU support.

Problem-size parameters visible in the code:

- number of simulations (config `split`: `train_ids`, `val_fraction`, sample caps)
- `training.batch_size`, `training.epochs`
- `training.max_train_batches`, `evaluation.max_val_batches`
- `patch_training` (contents TBC)

`train.py` reports "records" and "dataset items" separately, which suggests patch training can turn
one simulation into several training items. How compute, memory and time scale with the number of
simulations is **TBC from the owner**; this is needed before any weak-scaling claim.

Rubric applicability: core (host side), GPU (primary), I/O, and a limited intra-node study
(`num_workers` sweep; serial baseline `num_workers=0`). Inter-node and multi-GPU scaling are not
applicable.

## 5: Memory, storage and I/O

Memory: TBC. Input data is ~16 MB per simulation (~160 MB for 10). This is far below the guidebook's
20% of node memory; larger subsets will be needed for scaling studies. Map dimensions TBC
(`inspect_dataset.py` reports them).

Output per training epoch, from `train.py`:

- `history.csv` and `history.json`, rewritten in full
- `training_curves.png`, redrawn with matplotlib
- `latest.pt` checkpoint; `best.pt` when the monitored metric improves
- CPU-side peak-finding and matching over the whole validation set
- tqdm progress output to the terminal

There is no switch to disable these. For core and GPU measurements they should be off; for the I/O
rubric they stay on. A switch has been requested from the owner.

Other scripts in the workflow write evaluation CSV/JSON files, up to 20 overlay PNGs, per-sample
prediction files, and threshold-sweep plots.

Hard-coded default path: `inspect_dataset.py`, `visualize_batch.py` and `visualize_sample.py` default
`--root` to a user home directory; `--root` must always be passed.

## 6: Additional comments from submitter

- The project is at an early stage; there is no `requirements.txt` or `environment.yml` yet.
- The code has so far been run on the owner's laptop, which has a GPU.

## 7: Pre-assessment outcome

**Not yet decided.** Blocking items before the benchmark can be run on COSMA8:

1. Confirm the received archive contains `src/astrocluster/` and `configs/`.
2. Owner's environment export and PyTorch/CUDA version.
3. Agreement on the benchmark definition (training run on the data subset) and whether it is
   representative of intended production runs.

Once these are resolved, the next step is a timed training run on a COSMA8 GPU node to check the
Chapter 4 wrap-up criteria (builds and runs as a black box, runtime under 10 minutes best case,
memory footprint).
