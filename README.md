# Adaptive Implicit Filtering for Neural Distance Functions

Code and evaluation utilities for surface reconstruction from sparse and noisy 3D point clouds using neural signed distance functions (SDFs) and locally adaptive spatial-angular filtering, benchmarked on `abc_00_sparse` (Points2Surf ABC).

## Repository Layout

```text
configs/abc_00_sparse.yaml    # Experiment configuration for abc_00_sparse
scripts/setup_env.sh          # Python/trimesh setup and CUDA extension build check
scripts/run_eval.sh           # CLI wrapper for mesh evaluation
src/data/pointcloud.py        # .xyz loader, normalization, and k-NN neighbor indexing
src/models/sdf_network.py     # Geometric-initialized SDF MLP with skip connections
src/models/adaptive_filter.py # Locally adaptive spatial and angular k-NN filter
src/eval/metrics.py           # Chamfer Distance (L1/L2), Normal Consistency, F-Score
```

## Setup

Install Python dependencies and verify environment readiness:

```bash
./scripts/setup_env.sh
```

Install a CUDA-compatible `torch` build matching your GPU environment (e.g., Google Colab / Linux CUDA runtime) before running training or filtering modules.

## Evaluation

Compute surface reconstruction metrics between a predicted mesh and a ground-truth mesh:

```bash
./scripts/run_eval.sh path/to/pred_mesh.ply path/to/gt_mesh.ply 100000
```

## Attribution and Status

* **Baseline Reference:** Built on top of ideas and reference workflows from [`list17/ImplicitFiltering`](https://github.com/list17/ImplicitFiltering) for implicit neural surface reconstruction.
* **Research Status:** Active undergraduate research (CSE498R) extending implicit filtering with locally adaptive spatial and angular bandwidths on sparse/noisy point clouds.
