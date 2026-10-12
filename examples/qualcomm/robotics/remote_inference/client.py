

import logging
from typing import Optional
from executorch.examples.qualcomm.robotics.remote_inference.codec import MessagePackTensorCodec
from executorch.examples.qualcomm.robotics.transport.frame_socket import ConnectionConfig, FramedSocket


logger = logging.getLogger(__name__)



class RemoteClient:
    """Send and receive serialized messages through a framed TCP connection.

    This class owns the connection and composes:

    - MessagePackTensorCodec for object serialization.
    - FramedSocket for length-prefixed byte transport.

    It does not interpret the semantic meaning of messages.
    """

    def __init__(
        self,
        connection_config: ConnectionConfig,
        *,
        codec: Optional[MessagePackTensorCodec] = None,
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

        self._connection_config = connection_config
        self._codec = codec or MessagePackTensorCodec()

        self._read_timeout_seconds = read_timeout_seconds
        self._write_timeout_seconds = write_timeout_seconds

        self._transport: Optional[FramedSocket] = None
        self._closed = False

    def connect(self) -> None:
        """Establish the TCP connection if it is not connected."""

        if self._closed:
            raise RuntimeError(
                "Cannot connect a closed RemoteClient"
            )

        if self._transport is not None:
            return

        self._transport = FramedSocket.connect(
            self._connection_config,
            read_timeout_seconds=self._read_timeout_seconds,
            write_timeout_seconds=self._write_timeout_seconds,
        )
        print(
            f"[client] Connecting to remote server at {self._connection_config.host}:{self._connection_config.port}"
        )

    def send(self, message: object) -> None:
        """Serialize and send one message."""

        
        transport = self._require_transport()
        payload = self._codec.encode(message)
        transport.send(payload)
        print(
            f"[client] send message type={type(message).__name__}"
        )

    def recv(self) -> object:
        """Receive and deserialize one message."""

        transport = self._require_transport()
        payload = transport.recv()
        print(
            f"[client] Received payload payload_bytes={len(payload)}"
        )
        return self._codec.decode(payload)

    def exchange(self, message: object) -> object:
        """Send one message and synchronously receive its response."""

        self.send(message)
        return self.recv()

    def close(self) -> None:
        """Close the underlying connection."""

        if self._closed:
            return

        self._closed = True

        if self._transport is not None:
            self._transport.close()
            self._transport = None

    def _require_transport(self) -> FramedSocket:
        if self._closed:
            raise RuntimeError("RemoteClient is closed")

        if self._transport is None:
            raise RuntimeError(
                "RemoteClient is not connected"
            )

        return self._transport

    def __enter__(self) -> "RemoteClient":
        self.connect()
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self.close()
