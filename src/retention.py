# src/retention.py
from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

from sqlalchemy.orm import Session

from .db import RecognizedFace, UnknownFace
from .config import RetentionConfig, StorageConfig


logger = logging.getLogger(__name__)


def prune_old_faces(
    session: Session,
    retention_cfg: RetentionConfig,
    storage_cfg: StorageConfig,
) -> None:
    now = dt.datetime.utcnow()
    unknown_cutoff = now - dt.timedelta(days=retention_cfg.unknown_faces)
    recognized_cutoff = now - dt.timedelta(days=retention_cfg.recognized_faces)

    # Unknown faces
    old_unknown = (
        session.query(UnknownFace)
        .filter(UnknownFace.detected_at < unknown_cutoff)
        .all()
    )
    for u in old_unknown:
        _delete_file(storage_cfg.base_dir.parent / u.image_path)
        session.delete(u)

    # Recognized faces
    old_rec = (
        session.query(RecognizedFace)
        .filter(RecognizedFace.detected_at < recognized_cutoff)
        .all()
    )
    for r in old_rec:
        _delete_file(storage_cfg.base_dir.parent / r.image_path)
        session.delete(r)

    if old_unknown or old_rec:
        logger.info(
            "Pruned %d unknown and %d recognized faces",
            len(old_unknown),
            len(old_rec),
        )

    session.commit()


def _delete_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to delete %s: %s", path, exc)
