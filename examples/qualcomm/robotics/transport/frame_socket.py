"""

TCP byte transport

duty: 
connection
bytes setting
length-prefixed message
timeout
retry

TCP connect
TCP close
send bytes
receive bytes
4-byte message length framing
retry
timeout

"""

import struct
import logging
import socket
import time
from dataclasses import dataclass
from typing import Optional


logger = logging.getLogger(__name__)


_FRAME_HEADER = struct.Struct(">I")



@dataclass(frozen=True)
class ConnectionConfig:
    """Remote endpoint and bounded connection retry settings."""

    host: str
    port: int
    retry_timeout_seconds: float = 5.0
    retry_interval_seconds: float = 1.0

    def __post_init__(self) -> None:
        if not self.host:
            raise ValueError("host must be nonempty")
        if not 1 <= self.port <= 65535:
            raise ValueError(f"port must be in [1, 65535], got {self.port}")
        if self.retry_timeout_seconds <= 0:
            raise ValueError("retry_timeout_seconds must be positive")
        if self.retry_interval_seconds <= 0:
            raise ValueError("retry_interval_seconds must be positive")


class FramedSocket:
    """Length-prefixed binary connection over an established TCP socket."""

    def __init__(
        self,
        connection: socket.socket,
        read_timeout_seconds: float = 120.0,
        write_timeout_seconds: float = 120.0,
    ) -> None:
        if read_timeout_seconds <= 0:
            raise ValueError(
                "read_timeout_seconds must be positive"
            )

        if write_timeout_seconds <= 0:
            raise ValueError(
                "write_timeout_seconds must be positive"
            )

        self._connection = connection
        self.read_timeout_seconds = read_timeout_seconds
        self.write_timeout_seconds = write_timeout_seconds
        self._closed = False

    @classmethod
    def connect(
        cls,
        connection_config: ConnectionConfig,
        *,
        read_timeout_seconds: float = 120.0,
        write_timeout_seconds: float = 120.0,
    ) -> "FramedSocket":
        """Connect to a remote endpoint with bounded retries."""

        deadline = (
            time.monotonic()
            + connection_config.retry_timeout_seconds
        )
        last_error: Optional[OSError] = None

        while True:
            remaining = deadline - time.monotonic()

            if remaining <= 0:
                message = (
                    f"Unable to connect to "
                    f"{connection_config.host}:"
                    f"{connection_config.port} within "
                    f"{connection_config.retry_timeout_seconds:.3f}s"
                )

                if last_error is not None:
                    raise ConnectionError(
                        f"{message}: {last_error}"
                    ) from last_error

                raise ConnectionError(message)

            try:
                connection = socket.create_connection(
                    (
                        connection_config.host,
                        connection_config.port,
                    ),
                    timeout=remaining,
                )

                connection.setsockopt(
                    socket.IPPROTO_TCP,
                    socket.TCP_NODELAY,
                    1,
                )

                return cls(
                    connection,
                    read_timeout_seconds=read_timeout_seconds,
                    write_timeout_seconds=write_timeout_seconds,
                )

            except OSError as error:
                last_error = error
                remaining = deadline - time.monotonic()

                if remaining <= 0:
                    continue

                time.sleep(
                    min(
                        connection_config.retry_interval_seconds,
                        remaining,
                    )
                )
    
    def send(self, payload: bytes) -> None:
        """Send one complete framed byte payload."""

        self._ensure_open()

        payload_size = len(payload)
        self._connection.settimeout(self.write_timeout_seconds)

        self._connection.sendall(
            _FRAME_HEADER.pack(payload_size) + payload
        )

    def recv(self) -> bytes:
        """Receive one complete framed byte payload."""

        self._ensure_open()
        self._connection.settimeout(self.read_timeout_seconds)

        header = self._recv_exact(_FRAME_HEADER.size)
        (payload_size,) = _FRAME_HEADER.unpack(header)


        return self._recv_exact(payload_size)

    def _recv_exact(self, size: int) -> bytes:
        chunks: list[bytes] = []
        remaining = size

        while remaining > 0:
            
            chunk = self._connection.recv(remaining)
            chunks.append(chunk)
            remaining -= len(chunk)

        return b"".join(chunks)

    def _ensure_open(self) -> None:
        if self._closed:
            raise ConnectionError("FramedSocket is already closed")

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        try:
            self._connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        finally:
            self._connection.close()

    def __enter__(self) -> "FramedSocket":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

