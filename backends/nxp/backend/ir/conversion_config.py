# Copyright 2024-2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


class ConversionConfig:

    def __init__(self, args: dict | None = None):
        """
        Conversion configuration passed through command line arguments or gathered during
        the conversion process.

        :param args: Optional dictionary with conversion arguments. Unknown arguments are ignored.
        """
        self.keep_io_format: bool = False
        self.allow_inputs_stripping: bool = True
        self.qdq_aware_conversion: bool = True
        self.symbolic_dimensions_mapping: dict[str, int] | None = None
        self.input_shapes_mapping: dict[str, tuple] | None = None
        self.dont_skip_nodes_with_known_outputs: bool = False
        self.allow_select_ops: bool = True
        self.generate_artifacts_after_failed_shape_inference: bool = True

        self.optimization_whitelist: list | None = None
        self.optimization_blacklist: list | None = None

        self.non_negative_indices: bool = False
        self.cast_int64_to_int32: bool = False
        self.accept_resize_rounding_error: bool = False
        self.ignore_opset_version: bool = False

        self.tflite_quantization_integrity_check: bool = True

        if args is not None:
            for key, value in args.items():
                if key in self.__dict__:
                    setattr(self, key, value)

    def __repr__(self):
        attrs = []
        for attr in self.__dict__:
            attrs.append(f"{attr}={getattr(self, attr)}")

        return "ConversionConfig[" + ", ".join(attrs) + "]"


class QDQAwareConfig(ConversionConfig):

    def __init__(self):
        """
        Conversion config shortcut with QDQ aware conversion enabled.
        """
        super().__init__({"qdq_aware_conversion": True})
