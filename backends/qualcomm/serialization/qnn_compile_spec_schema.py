# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Please refer to executorch/backends/qualcomm/serialization/schema.fbs for the schema definitions
"""

from dataclasses import dataclass
from enum import IntEnum, unique


@unique
class HtpArch(IntEnum):
    NONE = 0
    V68 = 68
    V69 = 69
    V73 = 73
    V75 = 75


@dataclass
class HtpInfo:
    htp_arch: HtpArch = HtpArch.NONE
    vtcm_size_in_mb: int = 0


@unique
class QcomChipset(IntEnum):
    UNKNOWN_SM = 0
    SM8450 = 36  # v69
    SM8475 = 42  # v69
    SM8550 = 43  # v73
    SM8650 = 57  # v75


@dataclass
class SocInfo:
    soc_model: QcomChipset = QcomChipset.UNKNOWN_SM
    htp_info: HtpInfo = HtpInfo()


_soc_info_table = {
    QcomChipset.SM8450: SocInfo(QcomChipset.SM8450, HtpInfo(HtpArch.V69, 8)),
    QcomChipset.SM8475: SocInfo(QcomChipset.SM8475, HtpInfo(HtpArch.V69, 8)),
    QcomChipset.SM8550: SocInfo(QcomChipset.SM8550, HtpInfo(HtpArch.V73, 8)),
    QcomChipset.SM8650: SocInfo(QcomChipset.SM8650, HtpInfo(HtpArch.V75, 8)),
}


@unique
class QnnExecuTorchHtpPerformanceMode(IntEnum):
    kHtpDefault = 0
    kHtpSustainedHighPerformance = 1
    kHtpBurst = 2
    kHtpHighPerformance = 3
    kHtpPowerSaver = 4
    kHtpLowPowerSaver = 5
    kHtpHighPowerSaver = 6
    kHtpLowBalanced = 7
    kHtpBalanced = 8


@unique
class QnnExecuTorchHtpPrecision(IntEnum):
    kHtpQuantized = 0
    kHtpFp16 = 1


@unique
class QnnExecuTorchHtpPdSession(IntEnum):
    kHtpUnsignedPd = 0
    kHtpSignedPd = 1


@dataclass
class QnnExecuTorchHtpBackendOptions:
    performance_mode: QnnExecuTorchHtpPerformanceMode = (
        QnnExecuTorchHtpPerformanceMode.kHtpDefault
    )
    precision: QnnExecuTorchHtpPrecision = QnnExecuTorchHtpPrecision.kHtpQuantized
    pd_session: QnnExecuTorchHtpPdSession = QnnExecuTorchHtpPdSession.kHtpUnsignedPd
    use_conv_hmx: bool = True
    use_fold_relu: bool = True


@unique
class QnnExecuTorchBackendType(IntEnum):
    kUndefinedBackend = 0
    kGpuBackend = 1
    kHtpBackend = 2
    kDspBackend = 3


@unique
class QnnExecuTorchLogLevel(IntEnum):
    kLogOff = 0
    kLogLevelError = 1
    kLogLevelWarn = 2
    kLogLevelInfo = 3
    kLogLevelVerbose = 4
    kLogLevelDebug = 5


@dataclass
class QnnExecuTorchOptions:
    backend_type: QnnExecuTorchBackendType = QnnExecuTorchBackendType.kUndefinedBackend
    library_path: str = ""
    skel_library_dir: str = ""
    graph_name: str = ""
    log_level: QnnExecuTorchLogLevel = QnnExecuTorchLogLevel.kLogOff
    htp_options: QnnExecuTorchHtpBackendOptions = QnnExecuTorchHtpBackendOptions()
    soc_info: SocInfo = SocInfo()
    online_prepare: bool = False
    tensor_dump_output_path: str = ""
