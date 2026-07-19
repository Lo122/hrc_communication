"""UDP transport for sharing Events between local HRC processes."""

from __future__ import annotations

import json
import socket
from events import Event, EventType


class UDPEventSender:
    """Sends Events as compact JSON datagrams."""

    def __init__(self, host: str, port: int):
        self.address = (host, int(port))
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, event: Event) -> None:
        self._socket.sendto(json.dumps(event_to_dict(event), separators=(",", ":")).encode("utf-8"), self.address)

    def close(self) -> None:
        self._socket.close()


class UDPEventReceiver:
    """Receives Events from non-blocking UDP datagrams."""

    def __init__(self, host: str, port: int):
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.bind((host, int(port)))
        self._socket.setblocking(False)

    def poll(self) -> list[Event]:
        events = []
        while True:
            try:
                data, _ = self._socket.recvfrom(65535)
            except BlockingIOError:
                return events
            events.append(event_from_dict(json.loads(data.decode("utf-8"))))

    def close(self) -> None:
        self._socket.close()


def event_to_dict(event: Event) -> dict:
    return {
        "event_type": event.event_type.name,
        "source": event.source,
        "task_instance_id": event.task_instance_id,
        "payload": event.payload,
        "timestamp": event.timestamp,
    }


def event_from_dict(data: dict) -> Event:
    return Event(
        event_type=EventType[data["event_type"]],
        source=data["source"],
        task_instance_id=data.get("task_instance_id"),
        payload=data.get("payload") or {},
        timestamp=float(data["timestamp"]),
    )
