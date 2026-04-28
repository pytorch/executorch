# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

try:
    from qti.aisw.op_package_generator.generator import QnnOpPackageGenerator
except ImportError as e:
    raise ImportError(
        "Failed to import QnnOpPackageGenerator. "
        "Please run 'source $QNN_SDK_ROOT/bin/envsetup.sh' to set up the QNN SDK environment."
    ) from e

from executorch.backends.qualcomm.serialization.qc_schema import (
    QnnExecuTorchOpPackageInfo,
    QnnExecuTorchOpPackageOptions,
    QnnExecuTorchOpPackagePlatform,
    QnnExecuTorchOpPackageTarget,
)


class QnnCustomOpPackageBuilder:
    """
    Parses a QNN XML op package config and manages registration of
    target/platform/implementation for use with ExecuTorch.

    Validates that all keys in torch_op_name_map are present in the parsed
    package before any implementations are registered.
    """

    def __init__(
        self,
        xml_path: str,
        torch_op_name_map,
        interface_provider: Optional[str] = None,
    ):
        """
        Args:
            xml_path: Path to the QNN XML OpDef config file.
            torch_op_name_map: Maps QNN op type names to their corresponding
                PyTorch op targets.
                e.g. {"ExampleCustomOp": torch.ops.my_ops.custom_op.default}
            interface_provider: Interface provider symbol name. Defaults to
                "{PackageName}InterfaceProvider" if not specified.

        Raises:
            ValueError: If any key in torch_op_name_map is not found in the
                parsed op package.
        """
        op_package_generator = QnnOpPackageGenerator()
        op_package_generator.parse_config([xml_path])

        pkg_info = op_package_generator.package_infos[0]
        self.op_package_name = pkg_info.name
        self.interface_provider = (
            interface_provider
            if interface_provider
            else pkg_info.name + "InterfaceProvider"
        )
        self.torch_op_name_map = torch_op_name_map
        self._collection: List[QnnExecuTorchOpPackageInfo] = []
        self.operator_names = {op.type_name for op in pkg_info.operators}

        missing_ops = set()
        for qnn_op in self.torch_op_name_map.keys():
            if qnn_op not in self.operator_names:
                missing_ops.add(qnn_op)

        if len(missing_ops):
            raise ValueError(f"Ops missing from OpPackage: {missing_ops}")

    def register_implementation(
        self,
        target: QnnExecuTorchOpPackageTarget,
        platform: QnnExecuTorchOpPackagePlatform,
        op_package_path: str,
    ) -> "QnnCustomOpPackageBuilder":
        """
        Register one (target, platform, path) combination.
        Creates one QnnExecuTorchOpPackageInfo per op in torch_op_name_map.
        Returns self for method chaining.

        Args:
            target: QnnExecuTorchOpPackageTarget
            platform: QnnExecuTorchOpPackagePlatform
            op_package_path: Path to the implementation for the target/platform.
        """
        for qnn_op_type_name, torch_name in self.torch_op_name_map.items():
            self._collection.append(
                QnnExecuTorchOpPackageInfo(
                    op_package_name=self.op_package_name,
                    op_package_path=op_package_path,
                    interface_provider=self.interface_provider,
                    target=target,
                    custom_op_name=str(torch_name),
                    qnn_op_type_name=qnn_op_type_name,
                    platform=platform,
                )
            )
        return self

    def get_op_package_options(self) -> QnnExecuTorchOpPackageOptions:
        """
        Build and return QnnExecuTorchOpPackageOptions from all registered implementations.
        Call after all register_implementation() calls are complete.
        """
        options = QnnExecuTorchOpPackageOptions()
        options.op_package_infos = list(self._collection)
        return options
