#!/usr/bin/env bash
set -e

pip install -r requirements.txt

# Compile chamfer / custom CUDA extension if present
if [ -d "extensions/chamfer_dist" ]; then
    echo "Building chamfer CUDA extension..."
    cd extensions/chamfer_dist
    python setup.py install
    cd ../..
fi

# Quick sanity check for CUDA + trimesh
python -c "import torch, trimesh; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available(), 'trimesh:', trimesh.__version__)"
