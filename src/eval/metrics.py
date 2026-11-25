import argparse
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import trimesh


def sample_surface(mesh_path: str | Path, num_samples: int = 100000) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(mesh_path, force="mesh")
    points, face_idx = trimesh.sample.sample_surface(mesh, num_samples)
    normals = mesh.face_normals[face_idx]
    return points.astype(np.float32), normals.astype(np.float32)


def compute_metrics(
    pred_pts: np.ndarray,
    pred_normals: np.ndarray,
    gt_pts: np.ndarray,
    gt_normals: np.ndarray,
    fscore_thresh: float = 0.01,
) -> dict[str, float]:
    gt_tree = cKDTree(gt_pts)
    pred_tree = cKDTree(pred_pts)

    dist_p2g, idx_p2g = gt_tree.query(pred_pts, workers=-1)
    dist_g2p, idx_g2p = pred_tree.query(gt_pts, workers=-1)

    cd_l1 = 0.5 * (dist_p2g.mean() + dist_g2p.mean())
    cd_l2 = 0.5 * ((dist_p2g ** 2).mean() + (dist_g2p ** 2).mean())

    # Normal Consistency (unsigned dot product)
    nc_p2g = np.abs((pred_normals * gt_normals[idx_p2g]).sum(axis=-1)).mean()
    nc_g2p = np.abs((gt_normals * pred_normals[idx_g2p]).sum(axis=-1)).mean()
    nc = 0.5 * (nc_p2g + nc_g2p)

    # F-Score at threshold tau
    precision = (dist_p2g < fscore_thresh).mean()
    recall = (dist_g2p < fscore_thresh).mean()
    fscore = (2.0 * precision * recall) / (precision + recall + 1e-8)

    return {
        "CD_L1": float(cd_l1),
        "CD_L2": float(cd_l2),
        "NC": float(nc),
        "F-Score": float(fscore),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate reconstructed 3D mesh against ground truth.")
    parser.add_argument("--pred", required=True, help="Path to reconstructed mesh (.ply/.obj/.off)")
    parser.add_argument("--gt", required=True, help="Path to ground-truth mesh (.ply/.obj/.off)")
    parser.add_argument("--samples", type=int, default=100000, help="Surface sample count")
    parser.add_argument("--tau", type=float, default=0.01, help="F-Score distance threshold")
    args = parser.parse_args()

    p_pts, p_nrm = sample_surface(args.pred, args.samples)
    g_pts, g_nrm = sample_surface(args.gt, args.samples)
    res = compute_metrics(p_pts, p_nrm, g_pts, g_nrm, fscore_thresh=args.tau)

    for k, v in res.items():
        print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    main()
