# CARLA Perception: Multi-Modal 3D Object Detection

![Python](https://img.shields.io/badge/Python-3.7+-blue)
![CARLA](https://img.shields.io/badge/CARLA-0.9.x-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-1.x+-red)
![License](https://img.shields.io/badge/license-MIT-green)

A multi-modal perception pipeline for detecting vehicles, pedestrians, and cyclists in the [CARLA](https://carla.org/) autonomous driving simulator. Two detector implementations are provided — a pure LiDAR 3D detector and a sensor-fusion detector combining LiDAR with a 2D camera model — both evaluated using VOC Average Precision (AP).

---

## 📋 Overview

The project is structured around a `Detector` class interface that any detector must implement. The harness spawns the sensors defined by `Detector.sensors()`, feeds live sensor data into `Detector.detect()`, and scores the output against ground-truth annotations.

**Sensor Suite (both detectors):**
| Sensor | Config |
|--------|--------|
| Left RGB Camera | 1280×720, 90° FOV, 20 Hz |
| Right RGB Camera | 1280×720, 90° FOV, 20 Hz |
| 64-channel LiDAR | 50m range, 2.3M pts/sec, 20 Hz |
| GNSS / GPS | 5 Hz |

**Detected Object Classes:**
| ID | Class |
|----|-------|
| 0 | Pedestrian |
| 1 | Cyclist / Vehicle |
| 2 | Vehicle |
| 3 | Truck / Bus |

---

## 🗂️ Project Structure

```
.
├── lidar_detector.py            # LiDAR-only 3D detector (clustering + OBB fitting)
├── fusion_detector.py           # LiDAR 3D + Faster R-CNN 2D camera fusion detector
├── detector.py                  # Base Detector interface / sensor config
├── eval.py                      # VOC AP evaluation metrics
├── generate_traffic.py          # Spawns NPC traffic in CARLA
├── automatic_control.py         # Ego vehicle autopilot controller
├── agents/
│   ├── navigation/
│   │   ├── basic_agent.py
│   │   ├── behavior_agent.py
│   │   ├── local_planner.py
│   │   ├── global_route_planner.py
│   │   ├── controller.py
│   │   └── sensor_interface.py
│   └── tools/
│       └── misc.py
└── utils/
    ├── transform.py             # Coordinate frame transforms
    └── pygame_drawing.py        # Pygame visualization helpers
```

---

## 🔍 Detector Implementations

### 1. `lidar_detector.py` — LiDAR-Only 3D Detector

Processes raw LiDAR point clouds to produce 3D oriented bounding boxes in world coordinates. No camera input required.

**Pipeline:**
```
Raw LiDAR pts (N, 4)
    → Z-band filter (ground removal)
    → Voxel downsampling
    → 6-connected voxel clustering
    → OBB fitting per cluster
    → Size-based class assignment
    → Output: (N, 8, 3) world-frame boxes
```

**Key parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `lidar_voxel` | 0.35 m | Voxel grid resolution |
| `cluster_min_pts` | 35 | Min points to form a cluster |
| `zmin` / `zmax` | −1.0 / 3.0 m | Height band filter |
| `max_pts` | 120,000 | Max points per frame |

---

### 2. `fusion_detector.py` — LiDAR + Camera Fusion Detector

Extends the LiDAR detector with a **Faster R-CNN (ResNet-50 FPN)** 2D detector running on the Left camera stream. Both outputs are returned and can be used independently or combined by the evaluation harness.

**Additional pipeline (camera branch):**
```
Left camera RGBA (1280×720)
    → Convert to RGB uint8
    → Faster R-CNN (COCO pretrained)
    → Score threshold filter
    → Per-class NMS (IoU 0.5)
    → COCO → EE267 label remap
    → Output: (M, 4) 2D boxes (xyxy, image frame)
```

**COCO → class mapping:**
| COCO Label | Mapped Class |
|------------|-------------|
| person (1) | 0 – Pedestrian |
| bicycle (2), car (3), motorcycle (4), bus (6), truck (8) | 1 – Vehicle |

**Output keys:**

| Key | Shape | Description |
|-----|-------|-------------|
| `det_boxes` | `(N, 8, 3)` | 3D LiDAR boxes, world frame |
| `det_class` | `(N, 1)` | Class ID per 3D box |
| `det_score` | `(N, 1)` | Confidence per 3D box |
| `image_boxes` | `(M, 4)` | 2D camera boxes, xyxy image frame |
| `image_labels` | `(M,)` | Class ID per 2D box |
| `image_scores` | `(M,)` | Confidence per 2D box |

---

## 📐 Detector Interface

Both detectors implement the same interface, making them drop-in replaceable:

```python
class Detector:
    def sensors(self) -> list:
        """Return list of CARLA sensor specs to spawn."""
        ...

    def detect(self, sensor_data: dict, ego_pose_world=None, lidar_to_world=None) -> dict:
        """
        sensor_data keys: 'Left', 'Right', 'LIDAR', 'GPS'
        Each value: (frame_id, data_array)
        Returns dict with det_boxes, det_class, det_score (+ image_* for fusion).
        """
        ...
```

---

## 📊 Evaluation

Performance is scored using **VOC 2010 Average Precision** (AP), computed per class via IoU-based box matching using `shapely` polygon intersection (`eval.py`).

To run evaluation after an experiment:

```bash
python eval.py
```

---

## 🚀 Getting Started

### Prerequisites

1. **CARLA Simulator** (0.9.x)
   - Download: https://github.com/carla-simulator/carla/releases

2. **Python 3.7+**

3. **Install dependencies:**
   ```bash
   pip install numpy torch torchvision shapely pygame
   ```

4. **Add CARLA Python API to your path:**
   ```bash
   export PYTHONPATH=$PYTHONPATH:~/CARLA_0.9.13/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg
   ```

---

### Running

**Step 1 — Start CARLA:**
```bash
cd ~/CARLA_0.9.13
./CarlaUE4.sh        # Linux
# CarlaUE4.exe       # Windows
```

**Step 2 — (Optional) Spawn traffic:**
```bash
python generate_traffic.py
```

**Step 3 — Run the LiDAR detector:**
```bash
python lidar_detector.py
```

**Step 4 — Run the fusion detector:**
```bash
python fusion_detector.py
```

---

## 🛠️ Troubleshooting

**`No module named 'carla'`**
Add the CARLA egg to your `PYTHONPATH` (see Prerequisites).

**`Connection refused`**
The CARLA server must be fully loaded before running any client script.

**CUDA not available (fusion detector)**
The fusion detector falls back to CPU automatically. Expect slower inference (~2–5s per frame on CPU vs. ~100ms on GPU).

**Low AP scores**
- Check sensor transforms in `sensors()` match your CARLA vehicle setup
- Verify `lidar_to_world` transform is passed correctly into `detect()`
- Tune `cluster_min_pts` and `lidar_voxel` for your scene density

---

## 📚 References

- CARLA Simulator — *Dosovitskiy et al., CoRL 2017*
- Faster R-CNN — *Ren et al., NeurIPS 2015*
- VOC AP Metric — *Everingham et al., IJCV 2010*
- Eval code adapted from [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD)
