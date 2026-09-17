# MISFIT: Resolving Resource Contention in Dense Pervasive Edge Systems Beyond Memory Capacity

MISFIT is a GPU framework for **maintaining a maximal independent set (MIS)** on a large, continuously changing graph, instead of recomputing it from scratch after every change. Its base graph is kept as a flash-resident (mmap'd) binary CSR file rather than fully materialized in RAM, so the graph itself never has to fit entirely in host memory making it relevant on devices where the CPU and GPU share one physical memory pool like Jetson Orin or Nano. Incoming edge insertions/deletions are applied as an in-memory delta overlay on top of that base graph, and only the vertices actually affected by a change (and their bounded local neighborhood) are ever repaired.

This repository contains two variants of the same MIS-maintenance engine,
built and run identically except for one difference:

| Variant | Binary | Batch processing |
|---|---|---|
| `misfit` | `src/misfit/misfit.cu` | Strictly sequential: batch *K+1* doesn't start until batch *K*'s GPU work is finished and read back. |
| `misfit_async` | `src/misfit_async/misfit_async.cu` | Pipelined: batch *K+1*'s CPU-side neighborhood computation runs **while** batch *K*'s GPU kernels (already launched, on the same stream) are still executing. |

Both variants read the same `.egr` binary CSR graph format, the same initial-MIS seed format, and the same batch-folder format, and are invoked with the same 5 positional arguments (see [Running](#running))

## Requirements

- A CUDA-capable GPU and the CUDA toolkit (`nvcc`), tested against CUDA
  12.4. `nvcc` needs to support `-arch=native` (CUDA ≥ 11.6) or you'll
  need to pass your GPU's architecture explicitly (see
  [Building](#building)).
- A C++17-capable host compiler with OpenMP support (e.g. a reasonably
  recent `gcc`/`g++`).
- Python 3 with `numpy` for the scripts under `tools/` (graph conversion,
  batch generation, initial-MIS computation). Not needed to build or run
  the two binaries themselves.

## Cloning

```bash
git clone <this-repository-url> MISFIT
cd MISFIT
```

## Building

`run.sh` builds whichever variant you ask for automatically the first time
you run it (see [Running](#running)) — you don't need a separate build
step for normal use. To build both binaries directly instead:

```bash
nvcc -O3 -std=c++17 -Xcompiler -fopenmp -arch=native src/misfit/misfit.cu -o bin/misfit
nvcc -O3 -std=c++17 -Xcompiler -fopenmp -arch=native src/misfit_async/misfit_async.cu -o bin/misfit_async
```

`-arch=native` asks `nvcc` to target whatever GPU is installed on the build machine. On an older CUDA toolkit that doesn't support it, replace it with your GPU's explicit architecture flag instead, e.g. `-arch=sm_80` for an A100 or `-arch=sm_89` for an L40S/RTX 4090. `run.sh` picks this up
from the `MISFIT_ARCH_FLAG` environment variable (see below), so you don't need to edit the script to change it.

A binary at `bin/misfit` or `bin/misfit_async` is reused as-is by `run.sh` on later runs.

## Running

The quickest way to see MISFIT working end to end, using the small bundled sample dataset under `datasets/sample/`:

```bash
./run.sh --variant=misfit_async --sample
```

Against your own graph and batch stream:

```bash
./run.sh --variant=misfit_async \
    --graph=path/to/graph.egr \
    --mis=path/to/initial_mis.txt \
    --batch=path/to/batch_folder \
    --num-batches=10 \
    --gpu=0
```

- `--variant` —> `misfit` or `misfit_async` (see the table above).
- `--graph` —> the base graph, as a MISFIT `.egr` binary CSR file.
- `--mis` —> the initial MIS this run seeds from and then maintains
  incrementally: one 1-indexed vertex ID per line.
- `--batch` —> a folder of batch files named `1.mtx`, `2.mtx`, ... (see [Batch files](#batch-files) below), applied in that numeric order.
- `--num-batches` —> how many of those batch files to actually replay. Defaults to the number of `*.mtx` files found in `--batch` if omitted.
- `--gpu` —> CUDA device ID to run on. Defaults to `0`.

Run `./run.sh --help` for the full flag reference, including the
`NVCC`, `MISFIT_ARCH_FLAG`, and `MISFIT_SKIP_COMPACTION` environment
overrides.

You can also invoke either built binary directly, without `run.sh`, using
the same 5 positional arguments:

```bash
bin/misfit_async <graph.egr> <initial_mis.txt> <batch_folder> <num_batches> <gpu_id>
```

## Using your own graph

The `tools/` scripts take you from a plain-text edge list to everything
`run.sh` needs:

1. **Convert to `.egr`.** Your edge list must be a MatrixMarket-style
   `.mtx` file listing each undirected edge in **both** directions (`u v` and `v u`), one directed entry per line, 1-indexed. `datasets/sample/sample.mtx` is a working example.

   ```bash
   python3 tools/efficient_converter.py your_graph.mtx your_graph.egr
   ```

2. **Compute an initial MIS** to seed the incremental maintenance from:

   ```bash
   python3 tools/compute_initial_mis.py --egr your_graph.egr --out your_graph_mis.txt
   ```

3. **Generate batch files** — toggled insert/delete against current adjacency the first time each is replayed:

   ```bash
   python3 tools/generate_test_batches.py --egr your_graph.egr \
       --outdir your_batches --batch-size 2000 --num-batches 10 --seed 42
   ```


## File formats

**`.egr` binary CSR** (little-endian):
```
int32   nodes
int32   edge_count_or_sentinel   // -1 if the real count needs 64 bits
[int64  edge_count]              // present only if the above was -1
int32/int64  nindex[nodes + 1]   // int64 entries iff the sentinel was used
int32   nlist[edge_count]
```
`nindex[v] .. nindex[v+1]` indexes into `nlist` for vertex `v`'s neighbors.
An undirected edge (u, v) must appear as a directed entry in **both**
`nlist` rows.

**Initial MIS file**: plain text, one 1-indexed vertex ID per line, no header.

**Batch files**: plain text, named `1.mtx`, `2.mtx`, ... inside the batch folder (processed in that numeric order).

## Project layout

```
MISFIT/
├── run.sh                    # build (if needed) + run either variant
├── src/
│   ├── misfit/misfit.cu              # synchronous engine
│   └── misfit_async/misfit_async.cu  # pipelined (overlapped) engine
├── core/                     # shared headers used by both binaries
│   ├── FlashGraph.h          # mmap-backed .egr reader + delta overlay
│   ├── flash_compactor.h     # end-of-run base+delta -> fresh .egr merge
│   ├── gpu_common.h          # CUDA_CHECK, device setup, buffer allocation
│   ├── cli_setup.h           # argument parsing, batch-file I/O
│   └── reporting.h           # standardized console output
├── tools/                    # graph conversion / batch / MIS-seed utilities
├── datasets/sample/          # small bundled graph, initial MIS, and batches
└── bin/                      # build output (created by run.sh / nvcc)
```
