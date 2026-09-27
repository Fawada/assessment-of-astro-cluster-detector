# Assessment of astro-cluster-detector

Pre-assessment and high-level performance assessment of a neural-network galaxy cluster detector
(U-Net heatmap detector trained on weak-lensing convergence maps from DUSTGRAIN N-body simulations),
following the SHAREing performance assessment guidebook and the
[SHAREing assessment template](https://github.com/SHAREing-DRI/assessment-template).

**Status (2026-09-27): pre-assessment in progress.** No measurements have been taken on the
detector yet. See [`pre-assessment-report.md`](pre-assessment-report.md) for what is known,
what is missing, and what has been requested from the code owner.

## What is in this repository

| Path | Purpose |
|---|---|
| `pre-assessment-report.md` | Pre-assessment report (guidebook Chapter 4), based on the SHAREing template |
| `scripts/probe_node.sh` | Slurm job: characterises a COSMA8 GPU node (CPU, NUMA, GPU, modules, tools), measures CPU scalar peak with Likwid, checks data and GPU-counter access, and runs the GPU FLOP-count calibration |
| `scripts/ncu_calibrate.py` | Matmul and 3x3 convolution with analytically known FLOPs in FP32, TF32 and FP16, for checking which Nsight Compute counters capture tensor-core work |
| `results/` | Curated results (summaries and tables), added as measurements are made |

## What is deliberately not in this repository

- **The code under assessment.** It belongs to the code owner, was shared privately, and has no
  licence yet. It is kept outside this repo (or in the ignored `owner-code/` folder) until the owner
  agrees to how it is referenced.
- **Simulation data.** Kept on COSMA storage; never committed.
- **Raw profiler output.** Kept on COSMA; only summaries go into `results/`.

## Running the node probe on COSMA8

```bash
# On the login node: set up a Python environment with PyTorch (compute nodes may lack internet)
python3 -m venv ~/venvs/astro && source ~/venvs/astro/bin/activate && pip install torch

# Edit scripts/probe_node.sh: set --partition, --account, and the ENV SETUP lines
cd scripts && sbatch probe_node.sh
```

Results land in `probe_<jobid>/`, one text file per check.

> The probe scripts have not yet been run on COSMA8; the first run may need small fixes.

## Methodology note: GPU FLOP counting

The guidebook's GPU rubric (Section 5.3.2) counts CUDA-core FMA/ADD/MUL instructions via Nsight Compute.
This code trains a CNN with automatic mixed precision, so most arithmetic runs on tensor cores (FP16,
and TF32 by default for FP32 convolutions), which those counters do not include. `ncu_calibrate.py`
is used to quantify the gap and identify the tensor-pipe metrics and conversion factors needed, before
the rubric is applied to the detector. The denominator for C_GPU must then be the matching peak
(e.g. A100: ~19.5 TFLOP/s FP32, ~156 TF32 tensor, ~312 FP16 tensor, dense; confirm for the exact variant).

## Licence

Assessment material in this repository: MIT, following the SHAREing template. This does not apply to
the code under assessment.
