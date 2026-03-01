"""Application configuration with defaults."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path


def _project_root() -> Path:
    """Read-only bundle root (sys._MEIPASS when frozen, repo root in dev)."""
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parents[2]


def _user_data_dir() -> Path:
    """Writable data directory (%APPDATA%\\DetectorInventario when frozen)."""
    if getattr(sys, 'frozen', False):
        appdata = Path(os.environ.get('APPDATA', Path.home()))
        return appdata / 'DetectorInventario'
    return Path(__file__).resolve().parents[2] / 'data'


@dataclass
class Config:
    # Paths
    project_root: Path = field(default_factory=_project_root)
    data_dir: Path = field(default_factory=_user_data_dir)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "products.db"

    @property
    def images_dir(self) -> Path:
        return self.data_dir / "images"

    @property
    def pretrained_dir(self) -> Path:
        return self.data_dir / "models" / "pretrained"

    @property
    def classifier_dir(self) -> Path:
        return self.data_dir / "models" / "classifier"

    @property
    def embeddings_dir(self) -> Path:
        return self.data_dir / "models" / "embeddings"

    # Video processing
    frame_sample_fps: float = 2.0
    min_scene_change: float = 0.3

    # YOLO
    yolo_model: str = "yolov8n.pt"
    yolo_conf_threshold: float = 0.35
    yolo_iou_threshold: float = 0.45

    # SAM2
    sam2_model: str = "sam2_t.pt"
    use_sam2: bool = False

    # Background removal (GrabCut)
    remove_background: bool = True
    grabcut_iterations: int = 5

    # Cropping
    crop_padding: float = 0.10
    crop_size: int = 224

    # EfficientNet
    backbone_model: str = "efficientnet_v2_s"
    embedding_dim: int = 1280
    image_size: int = 224

    # Training
    train_epochs: int = 25
    train_lr: float = 1e-3
    train_batch_size: int = 32
    train_weight_decay: float = 0.01
    fine_tune_blocks: int = 3

    # FAISS / KNN
    knn_k: int = 5
    confidence_threshold: float = 0.45

    # Camera
    camera_fps: int = 30
    camera_width: int = 640
    camera_height: int = 480
    temporal_window: int = 5
    inference_skip_frames: int = 2

    def ensure_dirs(self) -> None:
        """Create all data directories if they don't exist."""
        for d in [
            self.data_dir,
            self.images_dir,
            self.pretrained_dir,
            self.classifier_dir,
            self.embeddings_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)


CONFIG = Config()
