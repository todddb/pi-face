# src/recognizer.py
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import face_recognition
from sqlalchemy.orm import Session

from .config import AppConfig
from .db import (
    get_session,
    KnownFaceEncoding,
    Person,
    RecognizedFace,
    UnknownFace,
)

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    top: int
    right: int
    bottom: int
    left: int
    person_id: Optional[int]
    person_name: Optional[str]
    distance: Optional[float]
    image_path: Path
    recognized: bool
    attributes: Dict[str, Any]


class FaceRecognizer:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self._known_encodings: Optional[np.ndarray] = None  # shape (N, 128)
        self._known_person_ids: List[int] = []
        self._last_seen: Dict[int, dt.datetime] = {}

    def load_known_faces(self, session: Session | None = None) -> None:
        own_session = False
        if session is None:
            session = get_session()
            own_session = True

        enc_rows = session.query(KnownFaceEncoding).all()
        logger.info("Loaded %d known face encodings from DB", len(enc_rows))

        encodings = []
        person_ids = []
        for row in enc_rows:
            if not row.encoding:
                continue
            arr = np.frombuffer(row.encoding, dtype=np.float32)
            if row.encoding_dim:
                try:
                    arr = arr.reshape((row.encoding_dim,))
                except ValueError:
                    logger.warning("Encoding %s has wrong dim; skipping", row.id)
                    continue
            encodings.append(arr)
            person_ids.append(row.person_id)

        if encodings:
            self._known_encodings = np.stack(encodings, axis=0)
            self._known_person_ids = person_ids
        else:
            self._known_encodings = None
            self._known_person_ids = []

        if own_session:
            session.close()

    def _encode_face(self, rgb_frame: np.ndarray, location: Tuple[int, int, int, int]):
        encodings = face_recognition.face_encodings(rgb_frame, [location])
        return encodings[0] if encodings else None

    def _find_best_match(self, embedding: np.ndarray) -> Tuple[Optional[int], Optional[float]]:
        if self._known_encodings is None or not len(self._known_person_ids):
            return None, None

        distances = face_recognition.face_distance(self._known_encodings, embedding)
        idx = np.argmin(distances)
        dist = float(distances[idx])
        if dist <= self.cfg.recognition.tolerance:
            return self._known_person_ids[idx], dist
        return None, dist

    def _estimate_attributes(self, face_bgr: np.ndarray) -> Dict[str, Any]:
        """Placeholder attribute estimator. Plug models in later."""
        if not self.cfg.attributes.enabled:
            return {}
        # TODO: Replace with actual age/emotion/glasses model (CPU or Hailo)
        return {
            "age": None,
            "gender": None,
            "emotion": None,
            "glasses": None,
        }

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        session: Session,
        now: Optional[dt.datetime] = None,
    ) -> List[DetectionResult]:
        if now is None:
            now = dt.datetime.utcnow()

        rgb_frame = frame_bgr[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        if not face_locations:
            return []

        results: List[DetectionResult] = []

        for (top, right, bottom, left) in face_locations:
            embedding = self._encode_face(rgb_frame, (top, right, bottom, left))
            if embedding is None:
                continue

            person_id, distance = self._find_best_match(embedding)

            h, w, _ = frame_bgr.shape
            top_clamped = max(0, top)
            bottom_clamped = min(h, bottom)
            left_clamped = max(0, left)
            right_clamped = min(w, right)
            face_crop = frame_bgr[top_clamped:bottom_clamped, left_clamped:right_clamped]

            attributes = self._estimate_attributes(face_crop)
            timestamp_str = now.strftime("%Y%m%d_%H%M%S_%f")

            if person_id is not None:
                person = session.query(Person).get(person_id)
                person_name = (
                    f"{person.first_name} {person.last_name or ''}".strip()
                    if person
                    else None
                )
                rel_path = Path("data/recognized_faces") / f"{person_id}_{timestamp_str}.jpg"
                self._save_image(face_crop, rel_path)

                rec = RecognizedFace(
                    person_id=person_id,
                    image_path=str(rel_path),
                    attributes=attributes or None,
                    distance=distance,
                    detected_at=now,
                )
                session.add(rec)
                logger.info(
                    "Recognized %s (id=%s) at distance=%.3f",
                    person_name or "unknown-name",
                    person_id,
                    distance or -1.0,
                )

                result = DetectionResult(
                    top=top_clamped,
                    right=right_clamped,
                    bottom=bottom_clamped,
                    left=left_clamped,
                    person_id=person_id,
                    person_name=person_name,
                    distance=distance,
                    image_path=rel_path,
                    recognized=True,
                    attributes=attributes,
                )
            else:
                rel_path = Path("data/unknown_faces") / f"unknown_{timestamp_str}.jpg"
                self._save_image(face_crop, rel_path)

                enc_bytes = embedding.astype(np.float32).tobytes()
                unknown = UnknownFace(
                    image_path=str(rel_path),
                    encoding=enc_bytes,
                    encoding_dim=int(embedding.shape[0]),
                    attributes=attributes or None,
                    detected_at=now,
                )
                session.add(unknown)
                logger.info("Stored unknown face at %s (distance=%.3f)", rel_path, distance or -1.0)

                result = DetectionResult(
                    top=top_clamped,
                    right=right_clamped,
                    bottom=bottom_clamped,
                    left=left_clamped,
                    person_id=None,
                    person_name=None,
                    distance=distance,
                    image_path=rel_path,
                    recognized=False,
                    attributes=attributes,
                )

            results.append(result)

        session.commit()
        return results

    def _save_image(self, image_bgr: np.ndarray, rel_path: Path) -> None:
        base = Path(__file__).resolve().parent.parent  # project root
        full = base / rel_path
        full.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(full), image_bgr)
