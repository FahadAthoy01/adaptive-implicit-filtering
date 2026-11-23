from pathlib import Path
from typing import Tuple
import numpy as np
from scipy.spatial import cKDTree
import torch
from torch.utils.data import Dataset


def load_xyz(path: str | Path) -> np.ndarray:
    pts = np.loadtxt(path, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(f"Expected (N, >=3) array in {path}, got {pts.shape}")
    return pts[:, :3]


def normalize_point_cloud(
    points: np.ndarray, scale_factor: float = 0.9
) -> Tuple[np.ndarray, np.ndarray, float]:
    centroid = points.mean(axis=0, keepdims=True)
    centered = points - centroid
    max_norm = float(np.linalg.norm(centered, axis=1).max())
    if max_norm < 1e-8:
        return centered, centroid.squeeze(0), 1.0
    normalized = (centered / max_norm) * scale_factor
    return normalized, centroid.squeeze(0), max_norm


class PointCloudDataset(Dataset):
    def __init__(
        self,
        xyz_path: str | Path,
        k_neighbors: int = 32,
        normalize: bool = True,
        scale_factor: float = 0.9,
    ) -> None:
        raw_pts = load_xyz(xyz_path)
        if normalize:
            self.points, self.centroid, self.scale = normalize_point_cloud(
                raw_pts, scale_factor=scale_factor
            )
        else:
            self.points = raw_pts
            self.centroid = np.zeros(3, dtype=np.float32)
            self.scale = 1.0

        self.k = min(k_neighbors, len(self.points))
        self.tree = cKDTree(self.points)
        # Precompute local neighbor indices for spatial/angular filtering
        _, self.knn_idx = self.tree.query(self.points, k=self.k)

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pt = torch.from_numpy(self.points[idx])
        nbrs = torch.from_numpy(self.points[self.knn_idx[idx]])
        return {"point": pt, "neighbors": nbrs}
