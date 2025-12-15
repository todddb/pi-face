# src/mqtt_client.py
from __future__ import annotations

import json
from typing import Any, Dict, Optional

import paho.mqtt.client as mqtt

from .config import MQTTConfig


class MQTTClient:
    def __init__(self, cfg: MQTTConfig):
        self.cfg = cfg
        self._client: Optional[mqtt.Client] = None
        if cfg.enabled:
            self._client = mqtt.Client()
            self._client.connect(cfg.host, cfg.port)

    def publish_event(self, topic_suffix: str, payload: Dict[str, Any]) -> None:
        if not self._client:
            return
        topic = f"{self.cfg.topic_prefix}/{topic_suffix}"
        self._client.publish(topic, json.dumps(payload), qos=0, retain=False)

    def loop(self) -> None:
        if self._client:
            self._client.loop(timeout=0.01)
