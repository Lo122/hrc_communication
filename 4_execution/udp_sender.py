"""UDP sender placeholder for Grasshopper communication."""

import socket

from config import UDP_HOST, UDP_PORT


class UDPSender:
    """Sends preformatted messages over UDP."""

    def __init__(self, host: str = UDP_HOST, port: int = UDP_PORT):
        self.host = host
        self.port = port
        # TODO: Reuse/import the existing UDP sender implementation if available.

    def send(self, message: str) -> None:
        """Send a message to Grasshopper."""
        # TODO: Add error handling and lifecycle management for production use.
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.sendto(message.encode("utf-8"), (self.host, self.port))
        print(f"[udp] {self.host}:{self.port} {message}")
