# src/config.py
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


BASE_DIR = Path(__file__).resolve().parent.parent  # ~/pi-face


@dataclass
class CameraConfig:
    resolution: tuple[int, int]
    framerate: int


@dataclass
class StorageConfig:
    base_dir: Path
    known_faces_dir: Path
    unknown_faces_dir: Path
    recognized_faces_dir: Path


@dataclass
class RetentionConfig:
    unknown_faces: int
    recognized_faces: int


@dataclass
class DatabaseConfig:
    url: str


@dataclass
class AttributesConfig:
    enabled: bool
    estimate_age: bool
    estimate_gender: bool
    estimate_emotion: bool
    detect_glasses: bool


@dataclass
class MQTTConfig:
    enabled: bool
    host: str
    port: int
    topic_prefix: str


@dataclass
class LoggingConfig:
    level: str
    file: Path


@dataclass
class RecognitionConfig:
    tolerance: float
    min_reencode_interval: float


@dataclass
class AppConfig:
    camera: CameraConfig
    storage: StorageConfig
    retention: RetentionConfig
    database: DatabaseConfig
    attributes: AttributesConfig
    mqtt: MQTTConfig
    logging: LoggingConfig
    recognition: RecognitionConfig


def _merge_defaults(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Very small helper to avoid KeyError if you later add fields."""
    return raw


def load_config(path: Path | None = None) -> AppConfig:
    if path is None:
        path = BASE_DIR / "config" / "config.yaml"

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    raw = _merge_defaults(raw)

    camera_cfg = raw.get("camera", {})
    storage_cfg = raw.get("storage", {})
    retention_cfg = raw.get("retention_days", {})
    db_cfg = raw.get("database", {})
    attr_cfg = raw.get("attributes", {})
    mqtt_cfg = raw.get("mqtt", {})
    log_cfg = raw.get("logging", {})
    rec_cfg = raw.get("recognition", {})

    storage_base = BASE_DIR / storage_cfg.get("base_dir", "./data")

    cfg = AppConfig(
        camera=CameraConfig(
            resolution=tuple(camera_cfg.get("resolution", [1280, 720])),
            framerate=int(camera_cfg.get("framerate", 15)),
        ),
        storage=StorageConfig(
            base_dir=storage_base,
            known_faces_dir=BASE_DIR / storage_cfg.get(
                "known_faces_dir", "./data/known_faces"
            ),
            unknown_faces_dir=BASE_DIR / storage_cfg.get(
                "unknown_faces_dir", "./data/unknown_faces"
            ),
            recognized_faces_dir=BASE_DIR / storage_cfg.get(
                "recognized_faces_dir", "./data/recognized_faces"
            ),
        ),
        retention=RetentionConfig(
            unknown_faces=int(retention_cfg.get("unknown_faces", 30)),
            recognized_faces=int(retention_cfg.get("recognized_faces", 3)),
        ),
        database=DatabaseConfig(
            url=db_cfg.get("url", "sqlite:///./pi_face.db"),
        ),
        attributes=AttributesConfig(
            enabled=bool(attr_cfg.get("enabled", True)),
            estimate_age=bool(attr_cfg.get("estimate_age", True)),
            estimate_gender=bool(attr_cfg.get("estimate_gender", True)),
            estimate_emotion=bool(attr_cfg.get("estimate_emotion", True)),
            detect_glasses=bool(attr_cfg.get("detect_glasses", True)),
        ),
        mqtt=MQTTConfig(
            enabled=bool(mqtt_cfg.get("enabled", False)),
            host=mqtt_cfg.get("host", "localhost"),
            port=int(mqtt_cfg.get("port", 1883)),
            topic_prefix=mqtt_cfg.get("topic_prefix", "pi-face"),
        ),
        logging=LoggingConfig(
            level=log_cfg.get("level", "INFO"),
            file=BASE_DIR / log_cfg.get("file", "./logs/pi-face.log"),
        ),
        recognition=RecognitionConfig(
            tolerance=float(rec_cfg.get("tolerance", 0.45)),
            min_reencode_interval=float(rec_cfg.get("min_reencode_interval", 2.0)),
        ),
    )
    return cfg
