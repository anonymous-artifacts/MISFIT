# run.sh -- build (if needed) and run one of MISFIT's two engines.
#
# Usage:
#   ./run.sh --variant=misfit|misfit_async --graph=<graph.egr> \
#       --mis=<initial_mis.txt> --batch=<batch_folder> \
#       [--num-batches=N] [--gpu=ID]
#
#   ./run.sh --variant=misfit_async --sample
#       (shortcut: fills in graph/mis/batch/num-batches from the small
#        bundled sample dataset under datasets/sample/)
#
# --num-batches defaults to the number of *.mtx files found in the batch
# folder if not given. --gpu defaults to 0.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat >&2 <<EOF
Usage: $0 --variant=misfit|misfit_async --graph=<graph.egr> --mis=<initial_mis.txt> --batch=<batch_folder> [--num-batches=N] [--gpu=ID]
       $0 --variant=misfit|misfit_async --sample

  --variant       misfit (synchronous engine) or misfit_async (overlapped CPU/GPU pipeline)
  --graph         path to the base graph, as a MISFIT .egr binary CSR file
  --mis           path to the initial MIS this run seeds from (one 1-indexed node ID per line)
  --batch         path to a folder of batch files named 1.mtx, 2.mtx, ... (see datasets/sample/batches)
  --num-batches   how many batch files to replay (default: count of *.mtx files in --batch)
  --gpu           CUDA device id to run on (default: 0)
  --sample        use the small bundled sample dataset instead of --graph/--mis/--batch

Environment overrides:
  NVCC              nvcc binary to build with (default: nvcc on PATH)
  MISFIT_ARCH_FLAG  nvcc architecture flag (default: -arch=native, needs CUDA >= 11.6;
                     on an older toolkit set this explicitly, e.g. MISFIT_ARCH_FLAG=-arch=sm_80)
  MISFIT_SKIP_COMPACTION=1   skip the end-of-run .egr compaction step (leaves the input graph file untouched)
EOF
}

VARIANT=""
GRAPH=""
MIS=""
BATCH=""
NUM_BATCHES=""
GPU="0"
USE_SAMPLE=0

for arg in "$@"; do
    case "$arg" in
        --variant=*)     VARIANT="${arg#--variant=}" ;;
        --graph=*)       GRAPH="${arg#--graph=}" ;;
        --mis=*)         MIS="${arg#--mis=}" ;;
        --batch=*)       BATCH="${arg#--batch=}" ;;
        --num-batches=*) NUM_BATCHES="${arg#--num-batches=}" ;;
        --gpu=*)         GPU="${arg#--gpu=}" ;;
        --sample)        USE_SAMPLE=1 ;;
        -h|--help)       usage; exit 0 ;;
        *)
            echo "ERROR: unrecognized argument: $arg" >&2
            usage
            exit 1
            ;;
    esac
done

if [ "$USE_SAMPLE" = "1" ]; then
    GRAPH="${GRAPH:-$SCRIPT_DIR/datasets/sample/sample.egr}"
    MIS="${MIS:-$SCRIPT_DIR/datasets/sample/sample_mis.txt}"
    BATCH="${BATCH:-$SCRIPT_DIR/datasets/sample/batches}"
fi

if [ -z "$VARIANT" ]; then
    echo "ERROR: --variant=misfit|misfit_async is required" >&2
    usage
    exit 1
fi
case "$VARIANT" in
    misfit)       SRC="$SCRIPT_DIR/src/misfit/misfit.cu";             BIN="$SCRIPT_DIR/bin/misfit" ;;
    misfit_async) SRC="$SCRIPT_DIR/src/misfit_async/misfit_async.cu"; BIN="$SCRIPT_DIR/bin/misfit_async" ;;
    *)
        echo "ERROR: --variant must be 'misfit' or 'misfit_async', got '$VARIANT'" >&2
        exit 1
        ;;
esac

if [ -z "$GRAPH" ] || [ -z "$MIS" ] || [ -z "$BATCH" ]; then
    echo "ERROR: --graph, --mis, and --batch are all required (or pass --sample)" >&2
    usage
    exit 1
fi
if [ ! -f "$GRAPH" ]; then
    echo "ERROR: graph not found: $GRAPH" >&2
    exit 1
fi
if [ ! -f "$MIS" ]; then
    echo "ERROR: initial MIS file not found: $MIS" >&2
    exit 1
fi
if [ ! -d "$BATCH" ]; then
    echo "ERROR: batch folder not found: $BATCH" >&2
    exit 1
fi

if [ -z "$NUM_BATCHES" ]; then
    NUM_BATCHES="$(find "$BATCH" -maxdepth 1 -name '*.mtx' | wc -l | tr -d ' ')"
    echo "--num-batches not given; found $NUM_BATCHES batch file(s) in $BATCH"
fi
case "$NUM_BATCHES" in
    ''|*[!0-9]*) echo "ERROR: num-batches must be a positive integer, got '$NUM_BATCHES'" >&2; exit 1 ;;
esac
case "$GPU" in
    ''|*[!0-9]*) echo "ERROR: --gpu must be a non-negative integer, got '$GPU'" >&2; exit 1 ;;
esac

NVCC="${NVCC:-nvcc}"
ARCH_FLAG="${MISFIT_ARCH_FLAG:--arch=native}"

mkdir -p "$SCRIPT_DIR/bin"
if [ -x "$BIN" ]; then
    echo "binary already built: $BIN (delete it to force a rebuild)"
else
    echo "building $VARIANT -> $BIN"
    echo "  $NVCC -O3 -std=c++17 -Xcompiler -fopenmp $ARCH_FLAG $SRC -o $BIN"
    "$NVCC" -O3 -std=c++17 -Xcompiler -fopenmp "$ARCH_FLAG" "$SRC" -o "$BIN"
fi

echo "=== running $VARIANT: graph=$GRAPH mis=$MIS batch=$BATCH num_batches=$NUM_BATCHES gpu=$GPU ==="
exec "$BIN" "$GRAPH" "$MIS" "$BATCH" "$NUM_BATCHES" "$GPU"
