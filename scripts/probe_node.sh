#!/bin/bash -l
# Pre-assessment probe for a COSMA8 GPU node (workbook Section 4.2 + GPU metric calibration).
#
# Before submitting, fill in:
#   --partition : GPU partition name  (list them with: sinfo -o "%P %G %N")
#   --account   : your COSMA project account
#   ENV SETUP   : modules / Python environment containing PyTorch (see below)
#
# Submit from the directory containing ncu_calibrate.py:
#   sbatch probe_node.sh
# Results land in probe_<jobid>/ (one text file per check) plus probe_<jobid>.out.

#SBATCH --job-name=probe_node
#SBATCH --partition=CHANGE_ME
#SBATCH --account=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --time=00:45:00
#SBATCH --output=probe_%j.out

# ---------------- ENV SETUP (edit) ----------------
# module purge
# module load <cuda module> <python module>
# source ~/venvs/astro/bin/activate
# --------------------------------------------------

set -u
OUT="probe_${SLURM_JOB_ID:-local}"
mkdir -p "$OUT"
DATA_ROOT="/home/gpetit/astro"
CALIB="$(pwd)/ncu_calibrate.py"

# run <name> <command...>: saves command + output to $OUT/<name>.txt, never aborts the job
run() {
    local name="$1"; shift
    local f="$OUT/$name.txt"
    echo "\$ $*" > "$f"
    if command -v "$1" >/dev/null 2>&1; then
        "$@" >> "$f" 2>&1 || echo "[exit status $?]" >> "$f"
    else
        echo "[not available: $1]" >> "$f"
    fi
    echo "  done: $name"
}

echo "== Machine characterisation =="
run hostname        hostname
run lscpu           lscpu
run numa            numactl -H
run memory          free -g
run nvidia_smi      nvidia-smi -q
run gpu_topology    nvidia-smi topo -m
run modules         module list
run likwid_topology likwid-topology

echo "== Tool availability =="
{
    for t in nvcc ncu nsys likwid-perfctr likwid-bench likwid-pin darshan-config darshan-parser valgrind py-spy python3; do
        printf "%-16s " "$t"; command -v "$t" || echo "NOT FOUND"
    done
    printf "%-16s " "/usr/bin/time"; [ -x /usr/bin/time ] && echo "/usr/bin/time" || echo "NOT FOUND"
} > "$OUT/tools.txt" 2>&1
run nvcc_version    nvcc --version
run ncu_version     ncu --version
run nsys_version    nsys --version
run darshan_logpath darshan-config --log-path

echo "== CPU scalar peak for the core rubric (workbook 5.2) =="
# Workbook convention: working set = half the L1 data cache. 16kB assumes a 32kB L1d;
# check lscpu.txt and adjust if different.
run likwid_peakflops likwid-bench -t peakflops -w S0:16kB:1

echo "== Access checks =="
{
    echo "Data root: $DATA_ROOT"
    ls -ld "$DATA_ROOT" 2>&1
    ls "$DATA_ROOT" 2>&1 | head -20
} > "$OUT/data_access.txt"
{
    echo "perf_event_paranoid: $(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo unknown)"
    grep -H . /proc/driver/nvidia/params 2>/dev/null | grep -i RmProfilingAdminOnly || echo "RmProfilingAdminOnly: not readable"
} > "$OUT/counter_permissions.txt"

echo "== PyTorch environment =="
python3 - > "$OUT/torch_env.txt" 2>&1 <<'EOF'
import sys
print("python", sys.version)
try:
    import torch
except ImportError as exc:
    print("torch not importable:", exc); raise SystemExit
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| cudnn", torch.backends.cudnn.version())
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("device:", p.name, "| capability:", f"{p.major}.{p.minor}",
          "| memory GB:", round(p.total_memory / 1e9, 1), "| SMs:", p.multi_processor_count)
    print("arch list compiled into torch:", torch.cuda.get_arch_list())
print("default cudnn.allow_tf32:", torch.backends.cudnn.allow_tf32)
print("default matmul.allow_tf32:", torch.backends.cuda.matmul.allow_tf32)
print("default cudnn.benchmark:", torch.backends.cudnn.benchmark)
print("torch CPU threads:", torch.get_num_threads())
EOF
echo "  done: torch_env"

echo "== GPU FLOP-count calibration =="
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null && [ -f "$CALIB" ]; then
    # Timing-only runs (no profiler)
    for op in mm conv; do
        for mode in fp32 tf32 fp16; do
            python3 "$CALIB" --op "$op" --mode "$mode" --iters 20 >> "$OUT/calib_timing.jsonl" 2>> "$OUT/calib_timing.err"
        done
    done
    echo "  done: calib_timing"

    if command -v ncu >/dev/null 2>&1; then
        # Metric names available on this GPU that relate to tensor cores
        ncu --query-metrics > "$OUT/ncu_all_metrics.txt" 2>&1
        grep -i -E "tensor|hmma" "$OUT/ncu_all_metrics.txt" > "$OUT/ncu_tensor_metrics.txt" 2>&1

        SASS='regex:^sm__sass_thread_inst_executed_op_(d|f|h)(fma|add|mul)_pred_on\.sum$'
        TENSOR='regex:^sm__inst_executed_pipe_tensor.*\.sum$'
        for op in mm conv; do
            for mode in fp32 tf32 fp16; do
                tag="${op}_${mode}"
                # Workbook recipe (CUDA-core FMA/ADD/MUL instructions)
                ncu --nvtx --nvtx-include "measure/" --metrics "$SASS" --csv \
                    --log-file "$OUT/ncu_sass_${tag}.csv" \
                    python3 "$CALIB" --op "$op" --mode "$mode" > "$OUT/ncu_sass_${tag}.json" 2>&1
                # Tensor-core pipe instructions
                ncu --nvtx --nvtx-include "measure/" --metrics "$TENSOR" --csv \
                    --log-file "$OUT/ncu_tensor_${tag}.csv" \
                    python3 "$CALIB" --op "$op" --mode "$mode" > "$OUT/ncu_tensor_${tag}.json" 2>&1
                echo "  done: ncu $tag"
            done
        done
        if grep -l -s "ERR_NVGPUCTRPERM" "$OUT"/ncu_*.csv "$OUT"/ncu_*.json >/dev/null; then
            echo "WARNING: ncu reported ERR_NVGPUCTRPERM - no access to GPU counters for normal users." \
                | tee "$OUT/NCU_PERMISSION_PROBLEM.txt"
        fi
    else
        echo "  ncu not found - skipped profiled calibration"
    fi
else
    echo "  PyTorch/CUDA not usable or $CALIB missing - skipped calibration"
fi

echo
echo "Probe complete. Results in $OUT/"
