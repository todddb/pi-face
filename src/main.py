# src/main.py
from __future__ import annotations

import argparse
import logging
import sys
import time
import datetime as dt

from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.db import init_db, get_session
from src.mqtt_client import MQTTClient
from src.recognizer import FaceRecognizer
from src.retention import prune_old_faces


def setup_logging(log_path: Path, level_str: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, level_str.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="pi-face real-time face recognition")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Capture a single frame and process it (for testing)",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV camera index (default 0)",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display video window with bounding boxes (debug)",
    )
    return parser.parse_args()


def main() -> None:
    cfg = load_config()
    setup_logging(cfg.logging.file, cfg.logging.level)

    logger = logging.getLogger("pi-face")
    logger.info("Starting pi-face")

    init_db(cfg.database.url)
    fr = FaceRecognizer(cfg)
    fr.load_known_faces()

    mqtt_client = MQTTClient(cfg.mqtt)

    args = parse_args()

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        logger.error("Failed to open camera index %d", args.camera_index)
        sys.exit(1)

    # Basic camera settings (may or may not have effect depending on backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.camera.resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera.resolution[1])
    cap.set(cv2.CAP_PROP_FPS, cfg.camera.framerate)

    last_retention_check = dt.datetime.utcnow()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Camera frame grab failed, retrying...")
                time.sleep(0.1)
                continue

            now = dt.datetime.utcnow()
            with get_session() as session:
                results = fr.process_frame(frame, session, now=now)

            for r in results:
                if r.recognized:
                    if r.ignored:
                        logger.debug(
                            "Ignoring notifications for %s (id=%s)",
                            r.person_name,
                            r.person_id,
                        )
                        continue

                    logger.debug(
                        "Recognized %s (id=%s) at %s",
                        r.person_name,
                        r.person_id,
                        r.image_path,
                    )
                    mqtt_client.publish_event(
                        "recognized",
                        {
                            "person_id": r.person_id,
                            "person_name": r.person_name,
                            "vip": None,  # can lookup vip flag later
                            "distance": r.distance,
                            "image_path": str(r.image_path),
                            "attributes": r.attributes,
                            "timestamp": now.isoformat(),
                        },
                    )
                else:
                    mqtt_client.publish_event(
                        "unknown",
                        {
                            "image_path": str(r.image_path),
                            "distance": r.distance,
                            "attributes": r.attributes,
                            "timestamp": now.isoformat(),
                        },
                    )

            # Simple retention check once per hour
            if (now - last_retention_check).total_seconds() > 3600:
                with get_session() as session:
                    prune_old_faces(session, cfg.retention, cfg.storage)
                last_retention_check = now

            mqtt_client.loop()

            if args.display:
                for r in results:
                    color = (0, 255, 0) if r.recognized else (0, 0, 255)
                    cv2.rectangle(
                        frame,
                        (r.left, r.top),
                        (r.right, r.bottom),
                        color,
                        2,
                    )
                    label = r.person_name or "Unknown"
                    cv2.putText(
                        frame,
                        label,
                        (r.left, r.top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                    )

                cv2.imshow("pi-face", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.once:
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        cap.release()
        if args.display:
            cv2.destroyAllWindows()
        logger.info("pi-face stopped")


if __name__ == "__main__":
    main()
