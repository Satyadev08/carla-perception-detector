# CARLA Part 2 - LiDAR-based detector
# Produces world-frame oriented 3D boxes from LiDAR point clouds.
# Contract:
#   Detector.sensors() -> list of sensor specs the harness will spawn
#   Detector.detect(sensor_data, ...) -> {
#       "det_boxes": (N, 8, 3) float32  # world coords
#       "det_class": (N, 1) int32       # 0=ped,1=cyclist,2=vehicle,3=truck/bus
#       "det_score": (N, 1) float32     # [0,1]
#   }

from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import math
import time


# ---------------------------
# Small linear algebra helpers
# ---------------------------

def _to_h(pts_xyz: np.ndarray) -> np.ndarray:
    """(N,3) -> (N,4) homogeneous."""
    if pts_xyz.size == 0:
        return np.zeros((0, 4), dtype=pts_xyz.dtype)
    ones = np.ones((pts_xyz.shape[0], 1), dtype=pts_xyz.dtype)
    return np.concatenate([pts_xyz, ones], axis=1)


def _apply_T(T: np.ndarray, pts_xyz: np.ndarray) -> np.ndarray:
    """Apply 4x4 transform to (N,3) -> (N,3)."""
    ph = _to_h(pts_xyz)  # (N,4)
    out = (T @ ph.T).T
    return out[:, :3]


# ---------------------------
# Filtering / Clustering
# ---------------------------

def _z_band(pts: np.ndarray, zmin: float, zmax: float) -> np.ndarray:
    """Keep points with z in [zmin, zmax]."""
    if pts.size == 0:
        return pts
    m = (pts[:, 2] >= zmin) & (pts[:, 2] <= zmax)
    return pts[m]


def _uniq_rows_int(A: np.ndarray):
    """Unique rows for int array; returns unique rows and indices kept."""
    _, idx = np.unique(A, axis=0, return_index=True)
    return idx


def _voxel_downsample(pts: np.ndarray, voxel: float) -> np.ndarray:
    """Hash-grid voxel downsample. Returns a subset of the input points."""
    if pts.size == 0:
        return pts
    keys = np.floor(pts / voxel).astype(np.int32)
    keep = _uniq_rows_int(keys)
    return pts[keep]


def _voxel_index_map(pts: np.ndarray, voxel: float) -> Tuple[np.ndarray, Dict[Tuple[int,int,int], List[int]]]:
    """Return integer voxel indices (N,3) and mapping from cell->point indices."""
    if pts.size == 0:
        return np.zeros((0,3), dtype=np.int32), {}
    ijk = np.floor(pts / voxel).astype(np.int32)
    buckets: Dict[Tuple[int,int,int], List[int]] = {}
    for i, cell in enumerate(ijk):
        key = (int(cell[0]), int(cell[1]), int(cell[2]))
        buckets.setdefault(key, []).append(i)
    return ijk, buckets


def _cluster_voxel_6conn(
    pts: np.ndarray,
    voxel: float,
    min_points_cluster: int,
    max_points_per_frame: int
) -> List[np.ndarray]:
    """
    Cluster points through 6-connected voxel adjacency.
    Returns list of clusters as raw point arrays (M_i, 3).
    """
    if pts.shape[0] == 0:
        return []

    # Cap for runtime
    if pts.shape[0] > max_points_per_frame:
        sel = np.random.choice(pts.shape[0], max_points_per_frame, replace=False)
        pts = pts[sel]

    ijk, buckets = _voxel_index_map(pts, voxel)
    visited = set()
    clusters: List[np.ndarray] = []

    for start_key in list(buckets.keys()):
        if start_key in visited:
            continue

        comp_vox = [start_key]
        queue = [start_key]
        visited.add(start_key)

        while queue:
            cx, cy, cz = queue.pop()
            for dx, dy, dz in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
                nk = (cx+dx, cy+dy, cz+dz)
                if nk in buckets and nk not in visited:
                    visited.add(nk)
                    queue.append(nk)
                    comp_vox.append(nk)

        # Gather points from the voxels in this component
        idxs = []
        for vk in comp_vox:
            idxs.extend(buckets[vk])

        if len(idxs) >= min_points_cluster:
            clusters.append(pts[np.asarray(idxs, dtype=np.int64)])

    return clusters


# ---------------------------
# Box fitting & labeling
# ---------------------------

def _pca_yaw_xy(pts: np.ndarray) -> float:
    """
    Estimate yaw (rotation about Z) from PCA on XY plane.
    Returns angle radians in (-pi, pi].
    """
    xy = pts[:, :2]
    mu = xy.mean(axis=0, keepdims=True)
    X = xy - mu
    # covariance
    cov = (X.T @ X) / max(1, xy.shape[0] - 1)
    evals, evecs = np.linalg.eigh(cov)
    main = evecs[:, np.argmax(evals)]
    return math.atan2(main[1], main[0])


def _fit_obb_world(cluster_pts_world: np.ndarray,
                   size_min=(0.3,0.3,0.5),
                   size_max=(12.0,4.0,4.5)) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Fit an oriented bounding box (center, LWH, yaw) in world frame.
    """
    n = cluster_pts_world.shape[0]
    if n < 3:
        c = cluster_pts_world.mean(axis=0)
        return c.astype(np.float32), np.array(size_min, dtype=np.float32), 0.0

    yaw = _pca_yaw_xy(cluster_pts_world)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[ cy, -sy, 0.0],
                   [ sy,  cy, 0.0],
                   [0.0, 0.0, 1.0]], dtype=np.float32)

    # Rotate into local frame to find axis-aligned extents
    pts_local = (Rz.T @ cluster_pts_world.T).T
    mins = pts_local.min(axis=0)
    maxs = pts_local.max(axis=0)
    size = (maxs - mins).astype(np.float32)
    size = np.maximum(size, np.array(size_min, dtype=np.float32))
    size = np.minimum(size, np.array(size_max, dtype=np.float32))

    # Local center then back to world; use centroid for stability
    c_world = cluster_pts_world.mean(axis=0).astype(np.float32)
    return c_world, size, float(yaw)


def _box_corners(center: np.ndarray, size_lwh: np.ndarray, yaw: float) -> np.ndarray:
    """
    Compute 8 corners for a box given center (3,), size (L,W,H) and yaw.
    Order is not enforced by spec.
    """
    L, W, H = size_lwh.tolist()
    dx, dy, dz = L/2.0, W/2.0, H/2.0

    # local corners
    local = np.array([
        [ dx,  dy,  dz],
        [ dx, -dy,  dz],
        [-dx, -dy,  dz],
        [-dx,  dy,  dz],
        [ dx,  dy, -dz],
        [ dx, -dy, -dz],
        [-dx, -dy, -dz],
        [-dx,  dy, -dz],
    ], dtype=np.float32)

    cy, sy = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[ cy, -sy, 0.0],
                   [ sy,  cy, 0.0],
                   [0.0, 0.0, 1.0]], dtype=np.float32)
    world = (Rz @ local.T).T + center[None, :]
    return world.astype(np.float32)


def _classify_size(size_lwh: np.ndarray) -> int:
    """
    Simple size-based class heuristic.
    0=pedestrian, 1=cyclist, 2=vehicle, 3=truck/bus
    """
    L, W, H = size_lwh.tolist()

    # pedestrian
    if L < 1.0 and W < 1.0 and 1.2 <= H <= 2.2:
        return 0

    # cyclist
    if 1.0 <= L <= 2.5 and 0.3 <= W <= 1.2 and 1.2 <= H <= 2.5:
        return 1

    # truck/bus (oversized)
    if L > 5.0 or W > 2.5 or H > 3.0:
        return 3

    # default vehicle
    return 2


def _score_from_count(npts: int) -> float:
    """Monotonic confidence ~ log(#pts) capped at 1."""
    return float(min(1.0, math.log1p(npts) / math.log1p(300.0)))


# ---------------------------
# Detector class (prof base)
# ---------------------------

class Detector:
    """
    LiDAR-only detector (clustering + OBB fitting).
    Exposes the expected interface:
      - sensors(): CARLA sensor spec list
      - detect(sensor_data, ...) -> dict with det_boxes/det_class/det_score
    """

    def __init__(self,
                 lidar_voxel: float = 0.35,
                 cluster_min_pts: int = 35,
                 zmin: float = -1.0,
                 zmax: float = 3.0,
                 max_pts: int = 120_000):
        self.voxel = lidar_voxel
        self.cluster_min_pts = cluster_min_pts
        self.zmin = zmin
        self.zmax = zmax
        self.max_pts = max_pts

    def sensors(self) -> List[Dict[str, Any]]:
        """
        Sensor blueprint/attribute hints are aligned with CARLA reference defaults.
        Names must match what your harness expects in sensor_data.
        """
        return [
            # Left RGB camera (1280x720, ~90° FOV)
            {
                "name": "Left",
                "bp": "sensor.camera.rgb",
                "transform": {"x": 1.5, "y": -0.5, "z": 2.0, "pitch": 0.0, "roll": 0.0, "yaw": -15.0},
                "attributes": {
                    "image_size_x": "1280",
                    "image_size_y": "720",
                    "fov": "90",
                    "sensor_tick": "0.05"
                }
            },
            # Right RGB camera
            {
                "name": "Right",
                "bp": "sensor.camera.rgb",
                "transform": {"x": 1.5, "y": 0.5, "z": 2.0, "pitch": 0.0, "roll": 0.0, "yaw": 15.0},
                "attributes": {
                    "image_size_x": "1280",
                    "image_size_y": "720",
                    "fov": "90",
                    "sensor_tick": "0.05"
                }
            },
            # LiDAR (ray-cast)
            {
                "name": "LIDAR",
                "bp": "sensor.lidar.ray_cast",
                "transform": {"x": 0.0, "y": 0.0, "z": 2.2, "pitch": 0.0, "roll": 0.0, "yaw": 0.0},
                "attributes": {
                    "channels": "64",
                    "points_per_second": "2300000",  # ~2.3M
                    "rotation_frequency": "20.0",    # 20 Hz
                    "upper_fov": "15.0",
                    "lower_fov": "-25.0",
                    "range": "50.0",
                    "sensor_tick": "0.05"
                }
            },
            # GNSS (useful to log)
            {
                "name": "GPS",
                "bp": "sensor.other.gnss",
                "transform": {"x": 0.0, "y": 0.0, "z": 2.0, "pitch": 0.0, "roll": 0.0, "yaw": 0.0},
                "attributes": {
                    "sensor_tick": "0.2"
                }
            }
        ]

    # -------- main API --------

    def detect(self,
               sensor_data: Dict[str, Any],
               ego_pose_world: Optional[np.ndarray] = None,
               lidar_to_world: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Run detection for the current frame.
        sensor_data: dict like
            {
              'Left':  (frame_id, image_np[...,3 or 4]),
              'Right': (frame_id, image_np),
              'LIDAR': (frame_id, pts (N,3 or N,4) [, T_world_from_lidar 4x4]),
              'GPS':   (frame_id, (lat,lon,alt) or dict)
            }
        Returns dict with det_* arrays.
        """
        t0 = time.time()

        if "LIDAR" not in sensor_data:
            return self._empty_out()

        pts_world = self._extract_lidar_world(sensor_data["LIDAR"], lidar_to_world)

        # Filter & downsample
        pts_world = _z_band(pts_world, self.zmin, self.zmax)
        if pts_world.shape[0] == 0:
            return self._empty_out()

        pts_ds = _voxel_downsample(pts_world, self.voxel * 0.7)

        clusters = _cluster_voxel_6conn(
            pts_ds,
            voxel=self.voxel,
            min_points_cluster=self.cluster_min_pts,
            max_points_per_frame=self.max_pts
        )

        boxes, labels, scores = [], [], []
        for cl in clusters:
            center, size_lwh, yaw = _fit_obb_world(
                cl,
                size_min=(0.3, 0.3, 0.5),
                size_max=(12.0, 4.0, 4.5)
            )

            size_lwh = np.clip(
                size_lwh,
                a_min=np.array([0.3, 0.3, 0.5], dtype=np.float32),
                a_max=np.array([12.0, 4.0, 4.5], dtype=np.float32)
            )

            cls_id = _classify_size(size_lwh)
            conf = _score_from_count(cl.shape[0])
            corners = _box_corners(center, size_lwh, yaw)

            boxes.append(corners)
            labels.append(cls_id)
            scores.append(conf)

        if len(boxes) == 0:
            return self._empty_out()

        det_boxes = np.stack(boxes, axis=0).astype(np.float32)           # (N,8,3)
        det_class = np.asarray(labels, dtype=np.int32).reshape(-1, 1)    # (N,1)
        det_score = np.asarray(scores, dtype=np.float32).reshape(-1, 1)  # (N,1)

        # Optionally log timing:
        # print(f"[Detector] {det_boxes.shape[0]} boxes in {1000*(time.time()-t0):.1f} ms")

        return {
            "det_boxes": det_boxes,
            "det_class": det_class,
            "det_score": det_score
        }

    # -------- helpers --------

    def _empty_out(self) -> Dict[str, np.ndarray]:
        return {
            "det_boxes": np.zeros((0, 8, 3), dtype=np.float32),
            "det_class": np.zeros((0, 1), dtype=np.int32),
            "det_score": np.zeros((0, 1), dtype=np.float32)
        }

    def _extract_lidar_world(self,
                             lidar_entry: Any,
                             fallback_T: Optional[np.ndarray]) -> np.ndarray:
        """
        Accepts:
          - (frame_id, Nx{3,4} [, 4x4 T_world_from_lidar])
          - Nx{3,4} directly
        Returns (N,3) world points using available transform or identity.
        """
        T = None
        pts = None

        if isinstance(lidar_entry, (tuple, list)):
            if len(lidar_entry) >= 2:
                pts = lidar_entry[1]
                if len(lidar_entry) >= 3:
                    T = np.asarray(lidar_entry[2], dtype=np.float32)
        else:
            pts = lidar_entry

        pts = np.asarray(pts)
        if pts.ndim != 2 or pts.shape[1] < 3:
            return np.zeros((0, 3), dtype=np.float32)

        pts_xyz = pts[:, :3].astype(np.float32)

        if T is None:
            T = fallback_T
        if T is not None:
            return _apply_T(T.astype(np.float32), pts_xyz)
        return pts_xyz  # assume already world-frame


# Optional factory if harness expects it
def build():
    return Detector()
