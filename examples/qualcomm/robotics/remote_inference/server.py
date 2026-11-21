"""
Synchronous object-level TCP server.

Responsibility boundary
-----------------------
RemoteServer owns:

- creating one TCP listener
- binding the listener to a configured host and port
- listening for incoming connections
- accepting one client connection
- wrapping the accepted connection with FramedSocket
- receiving and deserializing Python objects
- serializing and sending Python objects
- closing the accepted connection and listener

RemoteServer does not own:

- inference semantics
- quantization semantics
- policy execution
- model input or output validation
- request routing
- callbacks
- dataset selection
- rollout control
- calibration control

Lifecycle
---------
A RemoteServer instance accepts one client:

    accept()
    recv() / send()
    close()

The application script controls the communication sequence.

Example:

    with RemoteServer(host="127.0.0.1", port=5555) as server:
        server.accept()

        request = server.recv()
        response = application_logic(request)

        server.send(response)
        

Concurrency
-----------
RemoteServer is synchronous and not thread-safe. It supports one active client
connection and one send/receive operation at a time.
"""

from __future__ import annotations

import socket
from types import TracebackType

from executorch.examples.qualcomm.robotics.remote_inference.codec import (
    MessagePackTensorCodec,
)
from executorch.examples.qualcomm.robotics.transport.frame_socket import (
    FramedSocket,
)


class RemoteServer:
    """Exchange serialized Python objects with one TCP client."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        codec: MessagePackTensorCodec | None = None,
        read_timeout_seconds: float = 120.0,
        write_timeout_seconds: float = 120.0,
        listen_backlog: int = 1,
    ) -> None:
        if not host:
            raise ValueError(
                "host must be nonempty"
            )

        if not 1 <= port <= 65535:
            raise ValueError(
                f"port must be in [1, 65535], got {port}"
            )

        if read_timeout_seconds <= 0:
            raise ValueError(
                "read_timeout_seconds must be positive"
            )

        if write_timeout_seconds <= 0:
            raise ValueError(
                "write_timeout_seconds must be positive"
            )

        if listen_backlog <= 0:
            raise ValueError(
                "listen_backlog must be positive"
            )

        self._host = host
        self._port = port
        self._codec = (
            MessagePackTensorCodec()
            if codec is None
            else codec
        )

        self._read_timeout_seconds = read_timeout_seconds
        self._write_timeout_seconds = write_timeout_seconds
        self._listen_backlog = listen_backlog

        self._listener: socket.socket | None = None
        self._transport: FramedSocket | None = None
        self._client_address: tuple[str, int] | None = None

        self._closed = False

    @property
    def connected(self) -> bool:
        """Return whether a client connection is currently active."""

        return (
            self._transport is not None
            and not self._closed
        )

    @property
    def client_address(
        self,
    ) -> tuple[str, int] | None:
        """Return the connected client's address, if available."""

        return self._client_address

    def accept(self) -> None:
        """Create the listener and wait for one client connection.

        This method blocks until a client connects or the listener fails.

        A RemoteServer instance accepts only one client during its lifetime.
        """

        if self._closed:
            raise RuntimeError(
                "Cannot accept a connection on a closed RemoteServer"
            )

        if self._transport is not None:
            raise RuntimeError(
                "RemoteServer already has an active client connection"
            )

        if self._listener is not None:
            raise RuntimeError(
                "RemoteServer is already waiting for a client"
            )

        listener = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM,
        )

        try:
            listener.setsockopt(
                socket.SOL_SOCKET,
                socket.SO_REUSEADDR,
                1,
            )

            listener.bind(
                (self._host, self._port)
            )

            listener.listen(
                self._listen_backlog
            )

        except Exception:
            self._close_socket(listener)
            raise

        self._listener = listener

        print(
            f"[server] Listening "
            f"host={self._host} "
            f"port={self._port}",
            flush=True,
        )

        try:
            connection, address = listener.accept()

        except Exception:
            self._listener = None
            self._close_socket(listener)
            raise

        connection.setsockopt(
            socket.IPPROTO_TCP,
            socket.TCP_NODELAY,
            1,
        )

        try:
            transport = FramedSocket(
                connection,
                read_timeout_seconds=self._read_timeout_seconds,
                write_timeout_seconds=self._write_timeout_seconds,
            )

        except Exception:
            self._close_socket(connection)
            self._listener = None
            self._close_socket(listener)
            raise

        self._transport = transport
        self._client_address = address

        # This server accepts only one client, so the listener is no longer
        # needed after the connection has been established.
        self._listener = None
        self._close_socket(listener)

        print(
            f"[server] Client accepted "
            f"client={address[0]}:{address[1]}",
            flush=True,
        )

    def recv(self) -> object:
        """Receive and deserialize one complete Python object."""

        transport = self._require_transport()

        print(
            "[server] Waiting for message",
            flush=True,
        )

        payload = transport.recv()

        print(
            f"[server] Received payload "
            f"payload_bytes={len(payload)}",
            flush=True,
        )

        message = self._codec.decode(payload)

        print(
            f"[server] Message decoded "
            f"type={type(message).__name__}",
            flush=True,
        )

        return message

    def send(self, message: object) -> None:
        """Serialize and send one complete Python object."""

        transport = self._require_transport()

        payload = self._codec.encode(message)

        print(
            f"[server] Sending message "
            f"type={type(message).__name__} "
            f"payload_bytes={len(payload)}",
            flush=True,
        )

        transport.send(payload)

        print(
            f"[server] Message sent "
            f"type={type(message).__name__} "
            f"payload_bytes={len(payload)}",
            flush=True,
        )

    def exchange(self, message: object) -> object:
        """Send one object and synchronously receive one response.

        Most Server workflows use recv() followed by send(). This method is
        provided only for cases where the Server initiates an exchange after
        the connection has already been accepted.
        """

        self.send(message)
        return self.recv()

    def close(self) -> None:
        """Close the active connection and listener permanently."""

        if self._closed:
            return

        self._closed = True

        transport = self._transport
        self._transport = None
        self._client_address = None

        if transport is not None:
            print(
                "[server] Closing client connection",
                flush=True,
            )

            transport.close()

            print(
                "[server] Client connection closed",
                flush=True,
            )

        listener = self._listener
        self._listener = None

        if listener is not None:
            print(
                "[server] Closing listener",
                flush=True,
            )

            self._close_socket(listener)

            print(
                "[server] Listener closed",
                flush=True,
            )

    def _require_transport(self) -> FramedSocket:
        """Return the accepted client transport."""

        if self._closed:
            raise RuntimeError(
                "RemoteServer is closed"
            )

        if self._transport is None:
            raise RuntimeError(
                "RemoteServer has not accepted a client"
            )

        return self._transport

    @staticmethod
    def _close_socket(
        sock: socket.socket,
    ) -> None:
        """Best-effort shutdown and close of a raw socket."""

        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

        try:
            sock.close()
        except OSError:
            pass

    def __enter__(self) -> RemoteServer:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()