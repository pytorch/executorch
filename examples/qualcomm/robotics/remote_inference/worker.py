"""
Worker selection contract shared by Terminal A and Terminal B.

Terminal A is the master process. It selects one worker for each TCP
connection.

Terminal B is the passive worker process. It receives the selection,
acknowledges it, executes the requested work, closes the connection, and then
waits for another selection.

This module only defines worker selection. It does not own:

- TCP listener or connection lifecycle
- model inference
- rollout execution
- calibration execution
- dataset loading
- quantization
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum

from executorch.examples.qualcomm.robotics.remote_inference.client import (
    RemoteClient,
)
from executorch.examples.qualcomm.robotics.remote_inference.server import (
    RemoteServer,
)


WORKER_COMPLETE = "worker_complete"


class WorkerType(str, Enum):
    """Work supported by the passive Terminal B process."""

    ROLLOUT = "rollout"
    CALIBRATION = "calibration"
    SHUTDOWN = "shutdown"


def select_worker(
    server: RemoteServer,
    *,
    worker: WorkerType,
    name: str | None = None,
) -> None:
    """Select one Terminal B worker from Terminal A.

    Args:
        server:
            An accepted RemoteServer connection.

        worker:
            The worker Terminal B must execute.

        name:
            Required for rollout workers. It labels the rollout result and
            output video, for example ``exported`` or ``qdq``.

    This function sends the worker description and blocks until Terminal B
    acknowledges it.
    """

    if not isinstance(worker, WorkerType):
        raise TypeError(
            "worker must be a WorkerType, "
            f"got {type(worker)!r}"
        )

    if worker is WorkerType.ROLLOUT:
        if not isinstance(name, str) or not name:
            raise ValueError(
                "The rollout worker requires a nonempty name"
            )
    elif name is not None:
        raise ValueError(
            f"The {worker.value!r} worker does not accept a name"
        )

    description: dict[str, object] = {
        "worker": worker.value,
    }

    if name is not None:
        description["name"] = name

    print(
        f"[master] Selecting worker "
        f"worker={worker.value!r} "
        f"name={name!r}",
        flush=True,
    )

    server.send(description)

    acknowledged = server.recv()

    if acknowledged is not True:
        raise RuntimeError(
            "Terminal B did not acknowledge worker selection: "
            f"worker={worker.value!r}, name={name!r}"
        )

    print(
        f"[master] Worker acknowledged "
        f"worker={worker.value!r} "
        f"name={name!r}",
        flush=True,
    )


def receive_worker(
    client: RemoteClient,
) -> tuple[WorkerType, str | None]:
    """Receive and acknowledge the next worker selected by Terminal A.

    Returns:
        A tuple containing:

        - selected WorkerType
        - rollout name, or None for non-rollout workers
    """

    description = client.recv()

    if not isinstance(description, Mapping):
        raise TypeError(
            "Worker description must be a mapping, "
            f"got {type(description)!r}"
        )

    worker_value = description.get("worker")
    name = description.get("name")

    if not isinstance(worker_value, str):
        raise TypeError(
            "Worker description must contain a string 'worker'"
        )

    try:
        worker = WorkerType(worker_value)
    except ValueError as error:
        raise ValueError(
            f"Unsupported worker type: {worker_value!r}"
        ) from error

    unknown_fields = (
        set(description) - {"worker", "name"}
    )

    if unknown_fields:
        raise ValueError(
            "Unexpected worker description fields: "
            f"{sorted(unknown_fields)}"
        )

    if worker is WorkerType.ROLLOUT:
        if not isinstance(name, str) or not name:
            raise ValueError(
                "The rollout worker requires a nonempty name"
            )
    else:
        if name is not None:
            raise ValueError(
                f"The {worker.value!r} worker does not accept a name"
            )

        name = None

    print(
        f"[worker] Worker selected "
        f"worker={worker.value!r} "
        f"name={name!r}",
        flush=True,
    )

    client.send(True)

    return worker, name