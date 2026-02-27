import logging
from typing import List, Optional, Tuple

import torch
import torch_tensorrt

logger = logging.getLogger(__name__)


def multi_gpu_device_check() -> None:
    # If multi-device safe mode is disabled and more than 1 device is registered on the machine, warn user
    if (
        not torch_tensorrt.runtime._multi_device_safe_mode._PY_RT_MULTI_DEVICE_SAFE_MODE
        and torch.cuda.device_count() > 1
    ):
        logger.warning(
            "Detected this engine is being instantitated in a multi-GPU system with "
            "multi-device safe mode disabled. For more on the implications of this "
            "as well as workarounds, see the linked documentation "
            "(https://pytorch.org/TensorRT/user_guide/runtime.html#multi-device-safe-mode). "
            f"The engine is set to be instantiated on the current default cuda device, cuda:{torch.cuda.current_device()}. "
            "If this is incorrect, please set the desired cuda device via torch.cuda.set_device(...) and retry."
        )


def _is_switch_required(
    curr_device_id: int,
    engine_device_id: int,
    curr_device_properties: torch._C._CudaDeviceProperties,
    engine_device_properties: torch._C._CudaDeviceProperties,
) -> bool:
    """Determines whether a device switch is required based on input device parameters"""
    # Device Capabilities disagree
    if (curr_device_properties.major, curr_device_properties.minor) != (
        engine_device_properties.major,
        engine_device_properties.minor,
    ):
        logger.warning(
            f"Configured SM capability {(engine_device_properties.major, engine_device_properties.minor)} does not match with "
            f"current device SM capability {(curr_device_properties.major, curr_device_properties.minor)}. Switching device context."
        )

        return True

    # Names disagree
    if curr_device_properties.name != engine_device_properties.name:
        logger.warning(
            f"Program compiled for {engine_device_properties.name} but current CUDA device is "
            f"current device SM capability {curr_device_properties.name}. Attempting to switch device context for better compatibility."
        )

        return True

    # Device IDs disagree
    if curr_device_id != engine_device_id:
        logger.warning(
            f"Configured Device ID: {engine_device_id} is different than current device ID: "
            f"{curr_device_id}. Attempting to switch device context for better compatibility."
        )

        return True

    return False


def _select_rt_device(
    curr_device_id: int,
    engine_device_id: int,
    engine_device_properties: torch._C._CudaDeviceProperties,
) -> Tuple[int, torch._C._CudaDeviceProperties]:
    """Wraps compatible device check and raises error if none are found"""
    new_target_device_opt = _get_most_compatible_device(
        curr_device_id, engine_device_id, engine_device_properties
    )

    assert (
        new_target_device_opt is not None
    ), "Could not find a compatible device on the system to run TRT Engine"

    return new_target_device_opt


def _get_most_compatible_device(
    curr_device_id: int,
    engine_device_id: int,
    engine_device_properties: torch._C._CudaDeviceProperties,
) -> Optional[Tuple[int, torch._C._CudaDeviceProperties]]:
    """Selects a runtime device based on compatibility checks"""
    all_devices = [
        (i, torch.cuda.get_device_properties(i))
        for i in range(torch.cuda.device_count())
    ]
    logger.debug(f"All available devices: {all_devices}")
    target_device_sm = (engine_device_properties.major, engine_device_properties.minor)

    # Any devices with the same SM capability are valid candidates
    candidate_devices = [
        (i, device_properties)
        for i, device_properties in all_devices
        if (device_properties.major, device_properties.minor) == target_device_sm
    ]

    logger.debug(f"Found candidate devices: {candidate_devices}")

    # If less than 2 candidates are found, return
    if len(candidate_devices) <= 1:
        return candidate_devices[0] if candidate_devices else None

    # If more than 2 candidates are found, select the best match
    best_match = None

    for candidate in candidate_devices:
        i, device_properties = candidate
        # First priority is selecting a candidate which agrees with the current device ID
        # If such a device is found, we can select it and break out of the loop
        if device_properties.name == engine_device_properties.name:
            if i == curr_device_id:
                best_match = candidate
                break

            # Second priority is selecting a candidate which agrees with the target device ID
            # At deserialization time, the current device and target device may not agree
            elif i == engine_device_id:
                best_match = candidate

            # If no such GPU ID is found, select the first available candidate GPU
            elif best_match is None:
                best_match = candidate

    return best_match
