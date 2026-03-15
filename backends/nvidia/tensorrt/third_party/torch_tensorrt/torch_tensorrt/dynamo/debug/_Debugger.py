import contextlib
import functools
import logging
import os
import sys
import tempfile
from logging.config import dictConfig
from typing import Any, List, Optional
from unittest import mock

import torch
from torch_tensorrt._features import ENABLED_FEATURES
from torch_tensorrt.dynamo._defaults import DEBUG_LOGGING_DIR
from torch_tensorrt.dynamo.debug._DebuggerConfig import DebuggerConfig
from torch_tensorrt.dynamo.debug._supports_debugger import (
    _DEBUG_ENABLED_CLS,
    _DEBUG_ENABLED_FUNCS,
)
from torch_tensorrt.dynamo.lowering import (
    ATEN_POST_LOWERING_PASSES,
    ATEN_PRE_LOWERING_PASSES,
)

_LOGGER = logging.getLogger(__name__)
GRAPH_LEVEL = 5
logging.addLevelName(GRAPH_LEVEL, "GRAPHS")


class Debugger:
    def __init__(
        self,
        log_level: str = "debug",
        capture_fx_graph_before: Optional[List[str]] = None,
        capture_fx_graph_after: Optional[List[str]] = None,
        save_engine_profile: bool = False,
        capture_tensorrt_api_recording: bool = False,
        profile_format: str = "perfetto",
        engine_builder_monitor: bool = True,
        logging_dir: str = DEBUG_LOGGING_DIR,
        save_layer_info: bool = False,
    ):
        """Initialize a debugger for TensorRT conversion.

        Args:
            log_level (str): Logging level to use. Valid options are:
                'debug', 'info', 'warning', 'error', 'internal_errors', 'graphs'.
                Defaults to 'debug'.
            capture_fx_graph_before (List[str], optional): List of pass names to visualize FX graph
                before execution of a lowering pass. Defaults to None.
            capture_fx_graph_after (List[str], optional): List of pass names to visualize FX graph
                after execution of a lowering pass. Defaults to None.
            save_engine_profile (bool): Whether to save TensorRT engine profiling information.
                Defaults to False.
            capture_tensorrt_api_recording (bool): Whether to enable the capture TensorRT API recording feature, when this is enabled, it will output the catputure TensorRT API recording in the /tmp/torch_tensorrt_{current_user}/shim directory.
                It is part of the TensorRT capture and replay feature, the captured output will be able to replay for debug purpose.
                Defaults to False.
            profile_format (str): Format for profiling data. Choose from 'perfetto', 'trex', 'cudagraph'.
                If you need to generate engine graph using the profiling files, set it to 'trex' and use the C++ runtime.
                If you need to generate cudagraph visualization, set it to 'cudagraph'.
                Defaults to 'perfetto'.
            engine_builder_monitor (bool): Whether to monitor TensorRT engine building process.
                Defaults to True.
            logging_dir (str): Directory to save debug logs and profiles.
                Defaults to system temp directory.
            save_layer_info (bool): Whether to save layer info.
                Defaults to False.
        """

        os.makedirs(logging_dir, exist_ok=True)
        self.cfg = DebuggerConfig(
            log_level=log_level,
            save_engine_profile=save_engine_profile,
            capture_tensorrt_api_recording=capture_tensorrt_api_recording,
            engine_builder_monitor=engine_builder_monitor,
            logging_dir=logging_dir,
            profile_format=profile_format,
            save_layer_info=save_layer_info,
        )

        if log_level == "debug":
            self.log_level = logging.DEBUG
        elif log_level == "info":
            self.log_level = logging.INFO
        elif log_level == "warning":
            self.log_level = logging.WARNING
        elif log_level == "error":
            self.log_level = logging.ERROR
        elif log_level == "internal_errors":
            self.log_level = logging.CRITICAL
        elif log_level == "graphs":
            self.log_level = GRAPH_LEVEL

        else:
            raise ValueError(
                f"Invalid level: {log_level}, allowed levels are: debug, info, warning, error, internal_errors, graphs"
            )

        self.capture_fx_graph_before = capture_fx_graph_before
        self.capture_fx_graph_after = capture_fx_graph_after

        if self.cfg.capture_tensorrt_api_recording:
            if not sys.platform.startswith("linux"):
                _LOGGER.warning(
                    f"Capturing TensorRT API calls is only supported on Linux, therefore ignoring the capture_tensorrt_api_recording setting for {sys.platform}"
                )
            elif ENABLED_FEATURES.tensorrt_rtx:
                _LOGGER.warning(
                    "Capturing TensorRT API calls is not supported for TensorRT-RTX, therefore ignoring the capture_tensorrt_api_recording setting"
                )
            else:
                env_flag = os.environ.get("TORCHTRT_ENABLE_TENSORRT_API_CAPTURE", None)
                if env_flag is None or (env_flag != "1" and env_flag.lower() != "true"):
                    _LOGGER.warning(
                        "In order to capture TensorRT API calls, please invoke the script with environment variable TORCHTRT_ENABLE_TENSORRT_API_CAPTURE=1"
                    )
                _LOGGER.info("Capturing TensorRT API calls feature is enabled")

    def __enter__(self) -> None:
        self.original_lvl = _LOGGER.getEffectiveLevel()
        if ENABLED_FEATURES.torch_tensorrt_runtime:
            self.rt_level = torch.ops.tensorrt.get_logging_level()
        dictConfig(self.get_logging_config(self.log_level))

        if self.capture_fx_graph_before or self.capture_fx_graph_after:
            self.old_pre_passes, self.old_post_passes = (
                ATEN_PRE_LOWERING_PASSES.passes,
                ATEN_POST_LOWERING_PASSES.passes,
            )
            pre_pass_names = [p.__name__ for p in self.old_pre_passes]
            post_pass_names = [p.__name__ for p in self.old_post_passes]
            path = os.path.join(self.cfg.logging_dir, "lowering_passes_visualization")
            if self.capture_fx_graph_before is not None:
                pre_vis_passes = [
                    p for p in self.capture_fx_graph_before if p in pre_pass_names
                ]
                post_vis_passes = [
                    p for p in self.capture_fx_graph_before if p in post_pass_names
                ]
                ATEN_PRE_LOWERING_PASSES.insert_debug_pass_before(pre_vis_passes, path)
                ATEN_POST_LOWERING_PASSES.insert_debug_pass_before(
                    post_vis_passes, path
                )
            if self.capture_fx_graph_after is not None:
                pre_vis_passes = [
                    p for p in self.capture_fx_graph_after if p in pre_pass_names
                ]
                post_vis_passes = [
                    p for p in self.capture_fx_graph_after if p in post_pass_names
                ]
                ATEN_PRE_LOWERING_PASSES.insert_debug_pass_after(pre_vis_passes, path)
                ATEN_POST_LOWERING_PASSES.insert_debug_pass_after(post_vis_passes, path)

        self._context_stack = contextlib.ExitStack()

        for f in _DEBUG_ENABLED_FUNCS:
            f.__kwdefaults__["_debugger_config"] = self.cfg

        [
            self._context_stack.enter_context(
                mock.patch.object(
                    c,
                    "__init__",
                    functools.partialmethod(c.__init__, _debugger_config=self.cfg),
                )
            )
            for c in _DEBUG_ENABLED_CLS
        ]

    def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> None:

        dictConfig(self.get_logging_config(None))
        if ENABLED_FEATURES.torch_tensorrt_runtime:
            torch.ops.tensorrt.set_logging_level(self.rt_level)
        if self.capture_fx_graph_before or self.capture_fx_graph_after:
            ATEN_PRE_LOWERING_PASSES.passes, ATEN_POST_LOWERING_PASSES.passes = (
                self.old_pre_passes,
                self.old_post_passes,
            )
        self.debug_file_dir = tempfile.TemporaryDirectory().name

        for f in _DEBUG_ENABLED_FUNCS:
            f.__kwdefaults__["_debugger_config"] = None

        self._context_stack.close()

    def get_logging_config(self, log_level: Optional[int] = None) -> dict[str, Any]:
        level = log_level if log_level is not None else self.original_lvl
        config: dict[str, Any] = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "brief": {
                    "format": "%(asctime)s - %(levelname)s - %(message)s",
                    "datefmt": "%H:%M:%S",
                },
                "standard": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
            },
            "handlers": {
                "console": {
                    "level": level,
                    "class": "logging.StreamHandler",
                    "formatter": "brief",
                },
            },
            "loggers": {
                "": {  # root logger
                    "handlers": ["console"],
                    "level": level,
                    "propagate": True,
                },
            },
            "force": True,
        }
        if log_level is not None:
            config["handlers"]["file"] = {
                "level": level,
                "class": "logging.FileHandler",
                "filename": f"{self.cfg.logging_dir}/torch_tensorrt_logging.log",
                "formatter": "standard",
                "mode": "w",  # This will clear the previous content
            }
            config["loggers"][""]["handlers"].append("file")
        return config
