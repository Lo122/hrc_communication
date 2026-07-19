"""UDP sender for Grasshopper communication."""

import json
import socket
from typing import Any

from config import UDP_HOST, UDP_PORT


class UDPSender:
    """Sends preformatted messages over UDP."""

    def __init__(self, host: str = UDP_HOST, port: int = UDP_PORT):
        self.host = host
        self.port = int(port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, message: dict[str, Any] | str) -> None:
        """Send a JSON-compatible message or raw string to Grasshopper."""
        payload = self._encode_message(message)
        self.socket.sendto(payload, (self.host, self.port))
        print(f"[udp] {self.host}:{self.port} {payload.decode('utf-8')}")

    def close(self) -> None:
        """Close the UDP socket."""
        self.socket.close()

    @staticmethod
    def _encode_message(message: dict[str, Any] | str) -> bytes:
        if isinstance(message, str):
            return message.encode("utf-8")
        return json.dumps(message).encode("utf-8")
