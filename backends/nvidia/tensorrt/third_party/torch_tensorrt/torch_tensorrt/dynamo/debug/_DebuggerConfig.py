from dataclasses import dataclass

from torch_tensorrt.dynamo._defaults import DEBUG_LOGGING_DIR


@dataclass
class DebuggerConfig:
    log_level: str = "debug"
    save_engine_profile: bool = False
    capture_tensorrt_api_recording: bool = False
    engine_builder_monitor: bool = True
    logging_dir: str = DEBUG_LOGGING_DIR
    profile_format: str = "perfetto"
    save_layer_info: bool = False
