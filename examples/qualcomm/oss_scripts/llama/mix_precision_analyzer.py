# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import csv
import logging
import math
import re
import statistics
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field
from functools import reduce
from typing import Dict, List, Optional, Tuple

import torch
from executorch.backends.qualcomm.quantizer.quant_recipe import (
    ByNameRegex,
    ByNodeTarget,
    QuantGranularity,
    QuantRecipe,
)
from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
from executorch.devtools.inspector._intermediate_output_capturer import (
    IntermediateOutputCapturer,
)
from executorch.exir.debug_handle_utils import DEBUG_HANDLE_KEY
from torchao.quantization.pt2e import MinMaxObserver
from torchao.quantization.utils import compute_error


class PerLayerSqnrAnalyzer:
    """
    Computes per-layer SQNR by comparing fp32 and QDQ intermediate outputs.

    Args:
        model_name: Name of the model being analyzed.
        num_layers: Total number of transformer layers in the model.
        fp32_gm: fp32 exported GraphModule (before prepare_pt2e).
        qdq_gm: QDQ GraphModule (after convert_pt2e).
        analysis_recipe: The QuantRecipe used to produce qdq_gm. Stored in the
                returned SqnrReport and used as the baseline for diff annotation
                in save_suggest_recipes(). Pass None if the QDQ model was not
                produced via a QuantRecipe, in that case codegen will skip
                diff annotation entirely.
    """

    def __init__(
        self,
        model_name: str,
        num_layers: int,
        fp32_gm: torch.fx.GraphModule,
        qdq_gm: torch.fx.GraphModule,
        analysis_recipe: Optional[QuantRecipe] = None,
    ):
        self.model_name = model_name
        self.num_layers = num_layers
        self.fp32_gm = fp32_gm
        self.qdq_gm = qdq_gm
        self.analysis_recipe = analysis_recipe
        self.targets = {
            torch.ops.aten.conv2d.default,
        }
        self.q_ops = {
            torch.ops.torchao.quantize_affine,
            torch.ops.quantized_decomposed.quantize_per_channel.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
        }
        self.dq_ops = {
            torch.ops.torchao.dequantize_affine,
            torch.ops.quantized_decomposed.dequantize_per_channel.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
        }

    def analyze(self, samples: List[Tuple], num_sharding: int = 5) -> "SqnrReport":
        """
        Evaluates both the fp32 and QDQ graphs using the provided input_samples
        and computes the per-node Signal-to-Quantization-Noise Ratio (SQNR).

        Args:
            input_samples: A list of tuples containing tensors corresponding to the model's inputs.
            num_sharding: Number of contiguous layer groups to bucket the model into for SQNR
                aggregation. Rather than flagging individual layers, layers are grouped into
                ``num_sharding`` consecutive ranges (e.g. layers 0-7, 8-15, …) and the SQNR
                is averaged within each group. Because upgrading isolated layers is usually ineffective: quantization error from surrounding
                low-precision layers accumulates and dominates downstream behavior.

        Returns:
            An ``SqnrReport`` object containing the aggregated analysis results.
        """
        input_samples = [sample for sample in samples if sample is not None]

        if not input_samples:
            logging.warning("No input samples provided for analysis.")
            return SqnrReport(
                self.model_name, defaultdict(list), [], self.analysis_recipe
            )

        self._assign_debug_handles(self.fp32_gm)
        self._assign_debug_handles(self.qdq_gm)

        num_samples = len(input_samples)
        logging.info(f"num samples: {num_samples}")

        # Accumulate SQNR per module path across all input samples
        path_sqnr_sum = defaultdict(float)
        for sample in input_samples:
            fp_outputs = self._capture(self.fp32_gm, sample)
            qdq_outputs = self._capture(self.qdq_gm, sample)
            for path, sqnr in self._match_and_score(fp_outputs, qdq_outputs).items():
                path_sqnr_sum[path] += sqnr

        # Average the SQNRs and group them by normalized layer ranges
        report = defaultdict(list)
        for path, total_sqnr in path_sqnr_sum.items():
            group = self._normalize_group_name(
                path, self.num_layers, num_sharding=num_sharding
            )
            report[group].append(total_sqnr / num_samples)

        return SqnrReport(
            self.model_name,
            report,
            self._collect_conv_in_channels(),
            self.analysis_recipe,
        )

    def _assign_debug_handles(self, gm: torch.fx.GraphModule) -> None:
        call_nodes = []
        for node in gm.graph.nodes:
            if node.op != "call_function" or node.target not in self.targets:
                continue
            users = list(node.users.keys())
            maybe_q_op = users[0]
            if maybe_q_op.target in self.q_ops:
                maybe_q_op_users = list(maybe_q_op.users.keys())
                if maybe_q_op_users:
                    dq_node = maybe_q_op_users[0]
                    call_nodes.append(dq_node)
            else:
                call_nodes.append(node)

        for i, node in enumerate(call_nodes):
            node.meta[DEBUG_HANDLE_KEY] = i

    def _collect_conv_in_channels(self) -> List[int]:
        """Collects in_channels from all conv2d nodes in fp32_gm."""
        in_channels = []
        for node in self.fp32_gm.graph.nodes:
            if node.op == "call_function" and node.target in self.targets:
                weight_node = node.args[1]
                weight = weight_node.meta.get("val", None)
                if weight is not None:
                    in_channels.append(weight.shape[1])
        return in_channels

    def _capture(
        self, gm: torch.fx.GraphModule, inputs: Tuple
    ) -> Dict[int, Tuple[str, torch.Tensor]]:
        """
        Executes the graph module using IntermediateOutputCapturer and returns a mapping
        from debug_handle to a tuple of (module_path, output_tensor) for every captured tensor.
        """
        with torch.no_grad():
            raw = IntermediateOutputCapturer(gm).run_and_capture(*inputs)

        handle_idx_to_node = {
            n.meta[DEBUG_HANDLE_KEY]: n
            for n in gm.graph.nodes
            if DEBUG_HANDLE_KEY in n.meta
        }

        outputs: Dict[int, Tuple[str, torch.Tensor]] = {}
        for handle, tensor in raw.items():
            handle_idx = handle[0]
            if not isinstance(tensor, torch.Tensor):
                continue
            if handle_idx not in handle_idx_to_node:
                continue
            if path := self._module_path(handle_idx_to_node[handle_idx]):
                outputs[handle_idx] = (path, tensor)
        return outputs

    def _module_path(self, node: torch.fx.Node) -> str:
        if node.target in self.dq_ops:
            args = node.args
            node = args[0].args[0]
        if "nn_module_stack" in node.meta:
            return list(node.meta["nn_module_stack"].values())[-1][0]
        return None

    def _normalize_group_name(
        self, group_name: str, total_layers: int, num_sharding: int
    ) -> str:
        """Buckets layer indices into broader ranges for SQNR aggregation."""
        m = re.search(r"(layers)[_.](\d+)[_.]", group_name)
        if not m:
            return group_name
        prefix = m.group(1)
        layer_id = int(m.group(2))
        step = max(1, total_layers // num_sharding)
        idx = min(layer_id // step, num_sharding - 1)
        start = idx * step
        end = total_layers - 1 if idx == num_sharding - 1 else start + step - 1
        return re.sub(r"(layers)[_.]\d+[_.]", rf"{prefix}.[{start}-{end}].", group_name)

    def _match_and_score(
        self,
        fp_outputs: Dict[int, Tuple[str, torch.Tensor]],
        qdq_outputs: Dict[int, Tuple[str, torch.Tensor]],
    ) -> Dict[str, float]:
        """
        Compares corresponding fp32 and QDQ output tensors and computes their SQNR.
        Returns a dictionary mapping module paths to their SQNR values.
        """
        results = {}
        for handle, (path, fp_tensor) in fp_outputs.items():
            if handle in qdq_outputs and fp_tensor.dtype != torch.bool:
                _, qdq_tensor = qdq_outputs[handle]
                sqnr = compute_error(fp_tensor, qdq_tensor)
                if math.isfinite(sqnr):
                    results[path] = sqnr

        return results


@dataclass
class GroupSqnrStats:
    group_name: str
    avg_sqnr: float
    median_sqnr: float
    min_sqnr: float
    max_sqnr: float
    count: int


@dataclass
class SqnrReport:
    """Aggregated SQNR results from PerLayerSqnrAnalyzer."""

    model_name: str
    results: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    conv_in_channels: List[int] = field(default_factory=list)
    analysis_recipe: Optional[QuantRecipe] = field(default=None)

    def _compute_blk_sizes_candidate(
        self, min_size: int = 16, max_size: int = 64
    ) -> List[int]:
        """
        Derives block size candidates from the GCD of all conv in_channels,
        returning all divisors of that GCD in [min_size, max_size].
        Falls back to [min_size, max_size] if no channels are available.

        Empirically, block sizes in the range [16, 64] offer a good accuracy/compression
        trade-off for LPBQ quantization. Smaller values (e.g. 16) preserve more accuracy
        at the cost of larger model size; larger values (e.g. 64) compress more aggressively.
        You can widen the search range by adjusting ``min_size`` and ``max_size``.
        """
        if not self.conv_in_channels:
            return [min_size, max_size]
        gcd = reduce(math.gcd, self.conv_in_channels)
        return sorted(d for d in range(min_size, max_size + 1) if gcd % d == 0) or [
            min_size,
            max_size,
        ]

    def _group_to_regex(self, group_name: str) -> str:
        """
        Converts a normalized group name containing wildcards or bucketed intervals
        into a regular expression pattern suitable for ``QuantRecipe.add_regex()``.

        Examples:
            - ``layers.*.feed_forward.w2_conv`` -> ``r"layers\\..*\\.feed_forward\\.w2_conv"``
            - ``layers.[7-13].feed_forward`` -> ``r"layers\\.(7|8|9|10|11|12|13)\\.feed_forward"``
        """
        pattern = re.escape(group_name)
        # re.escape turns ".*." into "\.\*\." — restore the wildcard
        pattern = pattern.replace(r"\.\*\.", r"\..*\.")

        def expand_range(match):
            start, end = int(match.group(1)), int(match.group(2))
            return "(" + "|".join(str(i) for i in range(start, end + 1)) + ")"

        return re.sub(r"\\\[(\d+)\\-(\d+)\\\]", expand_range, pattern)

    def _group_stats(self) -> List[GroupSqnrStats]:
        stats = [
            GroupSqnrStats(
                group_name=grp,
                avg_sqnr=(sum(vals) / len(vals)).item(),
                median_sqnr=statistics.median(vals).item(),
                min_sqnr=min(vals).item(),
                max_sqnr=max(vals).item(),
                count=len(vals),
            )
            for grp, vals in self.results.items()
        ]
        stats.sort(key=lambda s: s.median_sqnr)
        return stats

    def save_analysis_summary(self, output_dir: Optional[str] = None) -> None:
        """Writes per-group SQNR statistics to ``{model_name}_quantization_error.csv``."""
        stats = self._group_stats()
        output_path = (
            f"{output_dir}/{self.model_name}_quantization_error.csv"
            if output_dir
            else f"{self.model_name}_quantization_error.csv"
        )
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "group_name",
                    "avg_sqnr",
                    "median_sqnr",
                    "min_sqnr",
                    "max_sqnr",
                    "count",
                ]
            )
            for s in stats:
                writer.writerow(
                    [
                        s.group_name,
                        s.avg_sqnr,
                        s.median_sqnr,
                        s.min_sqnr,
                        s.max_sqnr,
                        s.count,
                    ]
                )
        logging.info(
            f"SQNR analysis summary report saved to {self.model_name}_quantization_error.csv"
        )

    def suggest_recipe_overrides(
        self,
        blk_sizes_candidate: Optional[List[int]] = None,
        sqnr_threshold: float = 10.0,
        default_precision: QuantDtype = QuantDtype.use_16a4w_block,
        higher_precision: QuantDtype = QuantDtype.use_16a8w,
    ) -> List[QuantRecipe]:
        """
        Suggests precision upgrades on top of the recipe used during analysis.

        This function is intended to locate layer groups where the current quantization
        precision (as captured in ``report.analysis_recipe``) produces insufficient SQNR,
        and recommend upgrading only those groups to a higher precision. The suggested
        recipes are not standalone replacements — they are refinements of the analysis
        recipe, preserving its base structure while selectively elevating sensitive layers.

        Decision logic:
        - Groups with ``avg_sqnr``, ``median_sqnr``, or ``min_sqnr`` falling below
          ``sqnr_threshold`` are flagged as sensitive.
        - Sensitive groups are upgraded to ``higher_precision`` with PER_CHANNEL granularity.
        - Non-sensitive conv2d layers use ``default_precision`` with PER_BLOCK granularity,
          swept across multiple block size candidates.

        Args:
            blk_sizes_candidate: Block size candidates for non-sensitive layers. If ``None``
                                 (default), candidates are derived automatically from the GCD
                                 of all conv in_channels, keeping only divisors in [16, 64].
            sqnr_threshold: The SQNR threshold (in dB) below which a group is considered sensitive.
                            Defaults to 10.0 dB.
            default_precision: The base precision dtype for non-sensitive conv2d layers
                               (used with PER_BLOCK granularity).
            higher_precision: The elevated precision dtype for sensitive layers
                              (used with PER_CHANNEL granularity).

        Returns:
            A list of ``QuantRecipe`` objects, one per block size candidate, each representing
            a refined version of the analysis recipe with sensitive layers upgraded.
            Returns an empty list if no sensitive layers are detected.
        """
        if blk_sizes_candidate is None:
            blk_sizes_candidate = self._compute_blk_sizes_candidate()
            logging.info(
                f"[SqnrAnalyzer] Auto-derived blk_sizes_candidate: {blk_sizes_candidate}"
            )

        stats = self._group_stats()
        sensitive = [
            s
            for s in stats
            if s.avg_sqnr < sqnr_threshold
            or s.median_sqnr < sqnr_threshold
            or s.min_sqnr < sqnr_threshold
        ]

        if not sensitive:
            logging.info(
                "[SqnrAnalyzer] No sensitive layers detected. Keep the current configuration."
            )
            return []

        # Build keys for what the sensitive-layer pass will add, so we can skip
        # exact duplicates when copying analysis_recipe (sensitive replaces original
        # only when all four attributes match).
        sensitive_keys: set = {
            (
                frozenset({self._group_to_regex(s.group_name)}),
                higher_precision,
                QuantGranularity.PER_CHANNEL,
                (),
            )
            for s in sensitive
        }

        recipes: List[QuantRecipe] = []
        for blk_size in blk_sizes_candidate:
            recipe = QuantRecipe(
                QuantDtype.use_16a4w,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_TENSOR,
                verbose=True,
            ).add_node_target(
                {torch.ops.aten.conv2d.default},
                default_precision,
                False,
                act_observer=MinMaxObserver,
                granularity=QuantGranularity.PER_BLOCK,
                extra_kwargs={"block_size": (1, blk_size, 1, 1)},
                note="We use LPBQ for base precision",
            )

            # Carry over ByNameRegex strategies from analysis_recipe, skipping only
            # those that are identical to what the sensitive-layer pass will add
            # (same patterns, quant_dtype, granularity, and extra_kwargs).
            if self.analysis_recipe is not None:
                for strategy in self.analysis_recipe._strategies:
                    if isinstance(strategy, ByNameRegex):
                        key = (
                            frozenset(strategy.patterns),
                            strategy.quant_dtype,
                            strategy.granularity,
                            tuple(sorted(strategy.extra_kwargs.items())),
                        )
                        if key in sensitive_keys:
                            continue
                        recipe.add_regex(
                            strategy.patterns,
                            strategy.quant_dtype,
                            strategy.is_qat,
                            act_observer=strategy.act_observer,
                            granularity=strategy.granularity,
                            act_symmetric=strategy.act_symmetric,
                            extra_kwargs=strategy.extra_kwargs,
                            note=strategy.note,
                        )

            # Add sensitive layers with upgraded precision, replacing any original entry.
            for s in sensitive:
                pattern = self._group_to_regex(s.group_name)
                note = "[SqnrAnalyzer]:\n"
                if s.avg_sqnr < sqnr_threshold:
                    note += (
                        f" - avg_sqnr={s.avg_sqnr:.2f} < threshold={sqnr_threshold}\n"
                    )
                if s.median_sqnr < sqnr_threshold:
                    note += f" - median_sqnr={s.median_sqnr:.2f} < threshold={sqnr_threshold}\n"
                if s.min_sqnr < sqnr_threshold:
                    note += (
                        f" - min_sqnr={s.min_sqnr:.2f} < threshold={sqnr_threshold}\n"
                    )
                recipe.add_regex(
                    {pattern},
                    higher_precision,
                    False,
                    act_observer=MinMaxObserver,
                    granularity=QuantGranularity.PER_CHANNEL,
                    note=note,
                )
            recipes.append(recipe)

        return recipes


def save_suggest_recipes(  # noqa: C901
    report: "SqnrReport",
    suggest_recipe: List[QuantRecipe],
    output_dir: Optional[str] = None,
) -> None:
    """
    Generates and saves a Python script containing quantization recipe classes
    based on the suggested QuantRecipe objects from ``SqnrReport.suggest_recipe_overrides()``.

    The baseline recipe for diff annotation is taken from ``report.analysis_recipe`` — the
    ``QuantRecipe`` used to produce the QDQ model during analysis. If ``report.analysis_recipe``
    is None (i.e. the QDQ model was not produced via a QuantRecipe), diff annotation is skipped
    and all add_regex calls are emitted without [Original recipe] / [Added by SqnrAnalyzer] tags.

    Args:
        report: The ``SqnrReport`` returned by ``PerLayerSqnrAnalyzer.analyze()``.
                Provides ``model_name`` and ``analysis_recipe`` (the analysis-time baseline).
        suggest_recipe: List of QuantRecipe objects, from
                        ``SqnrReport.suggest_recipe_overrides()``.
    """
    if not suggest_recipe:
        logging.info(
            "There are no sensitive layers detected. You may keep your current configuration."
        )
        return

    model_name = report.model_name
    analysis_recipe = report.analysis_recipe
    class_name_prefix = model_name.upper().replace("-", "_")
    output_path = (
        f"{output_dir}/{model_name}_suggest_recipe.py"
        if output_dir
        else f"{model_name}_suggest_recipe.py"
    )

    file_header = textwrap.dedent(
        """\
        # Auto-generated by save_suggest_recipes()
        #
        # These recipes are REFINEMENTS of the recipe used during SQNR analysis.
        # They preserve the base quantization structure and selectively upgrade
        # layer groups where SQNR fell below the configured threshold.
        #
        # Each class below corresponds to a different LPBQ block size for the base layers.
        # Review the [Original recipe] / [Added by SqnrAnalyzer] annotations to understand
        # what changed relative to the analysis-time recipe, then pick the variant that
        # gives the best accuracy / model-size trade-off on your target device.

        import torch
        from executorch.backends.qualcomm.quantizer.custom_annotation import annotate_kv_8bit
        from executorch.backends.qualcomm.quantizer.quant_recipe import (
            QuantGranularity,
            QuantRecipe,
        )
        from executorch.backends.qualcomm.quantizer.quantizer import QuantDtype
        from torchao.quantization.pt2e import MinMaxObserver
        from examples.qualcomm.oss_scripts.llama.static_llm_quant_recipe import StaticLLMQuantRecipe
    """
    )

    # Collect existing strategies from the analysis recipe for diff annotation.
    # A strategy is [Original recipe] only if patterns/targets AND quant_dtype,
    # granularity, extra_kwargs all match an existing strategy in analysis_recipe.
    # If analysis_recipe is None, diff annotation is skipped entirely.
    original_regex_keys: set = set()
    original_target_keys: set = set()
    if analysis_recipe is not None:
        for strategy in analysis_recipe._strategies:
            if isinstance(strategy, ByNameRegex):
                key = (
                    frozenset(strategy.patterns),
                    strategy.quant_dtype,
                    strategy.granularity,
                    tuple(sorted(strategy.extra_kwargs.items())),
                )
                original_regex_keys.add(key)
            elif isinstance(strategy, ByNodeTarget):
                key = (
                    frozenset(strategy.targets),
                    strategy.quant_dtype,
                    strategy.granularity,
                    tuple(sorted(strategy.extra_kwargs.items())),
                )
                original_target_keys.add(key)

    generated_classes: List[str] = []
    recipe_classes: List[str] = []

    for recipe in suggest_recipe:
        node_target_strategies = [
            s for s in recipe._strategies if isinstance(s, ByNodeTarget)
        ]
        assert node_target_strategies and node_target_strategies[0].extra_kwargs.get(
            "block_size"
        ), "Expected at least one LPBQ node target strategy for PTQ recipes"
        blk_size = node_target_strategies[0].extra_kwargs["block_size"][1]
        class_name = f"{class_name_prefix}_BlockSize{blk_size}QuantRecipe"
        generated_classes.append(class_name)

        # Prepend [Original recipe] / [Added by SqnrAnalyzer] tag to each strategy's
        # note before codegen, then restore. Only done when analysis_recipe is set.
        saved_notes = {}
        if analysis_recipe is not None:
            for strategy in recipe._strategies:
                if isinstance(strategy, ByNameRegex):
                    key = (
                        frozenset(strategy.patterns),
                        strategy.quant_dtype,
                        strategy.granularity,
                        tuple(sorted(strategy.extra_kwargs.items())),
                    )
                    tag = (
                        "[Original recipe]\n"
                        if key in original_regex_keys
                        else "[Added by SqnrAnalyzer]\n"
                    )
                elif isinstance(strategy, ByNodeTarget):
                    key = (
                        frozenset(strategy.targets),
                        strategy.quant_dtype,
                        strategy.granularity,
                        tuple(sorted(strategy.extra_kwargs.items())),
                    )
                    tag = (
                        "[Original recipe]\n"
                        if key in original_target_keys
                        else "[Added by SqnrAnalyzer]\n"
                    )
                else:
                    continue
                saved_notes[id(strategy)] = strategy.note
                strategy.note = tag + strategy.note

        recipe_body = recipe.to_source()

        for strategy in recipe._strategies:
            if id(strategy) in saved_notes:
                strategy.note = saved_notes[id(strategy)]

        indent = "\t"
        init_body = (
            "super().__init__()\n"
            "\n"
            "self.recipe = " + recipe_body.lstrip() + "\n"
            "self.recipe.custom_quant_annotations.append(annotate_kv_8bit)"
        )
        class_body = (
            f"default_quant_dtype = QuantDtype.{recipe._default_quant_dtype.name}\n"
            "\n"
            "def __init__(self, verbose: bool = False):\n"
            + textwrap.indent(init_body, indent)
        )
        recipe_class = (
            f"class {class_name}(StaticLLMQuantRecipe):\n"
            + textwrap.indent(class_body, indent)
            + "\n"
        )
        recipe_classes.append(recipe_class)

    class_list = "\n".join(f"#        {cls}" for cls in generated_classes)
    usage_comments = (
        textwrap.dedent(
            """\
        # 
        # HOW TO USE THESE RECIPES
        # 
        #
        # The classes above were generated by the SQNR analyzer.
        # Each variant uses a different LPBQ block size for the base layers
        # while upgrading sensitive layers to higher precision.
        #
        # Suggested steps:
        #   1. Pick one class to try:
        """
        )
        + class_list
        + textwrap.dedent(
            f"""
        #
        #   2. In your export script, replace the original recipe import, e.g.:
        #        # Before:
        #        from examples.qualcomm.oss_scripts.llama.static_llm_quant_recipe import \\
        #            {class_name_prefix}QuantRecipe
        #        # After (example with Blk32):
        #        from <this_file_module> import {class_name_prefix}Blk32QuantRecipe as {class_name_prefix}QuantRecipe
        #
        #   3. Run calibration + export and compare perplexity / accuracy.
        #   4. If accuracy is still insufficient, try a smaller block size
        #      or increase the SQNR threshold and re-run the analyzer.
        """
        )
    )

    lines: List[str] = (
        file_header + "\n" + "\n".join(recipe_classes) + "\n" + usage_comments
    ).splitlines()

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    logging.info(f"\n[SqnrAnalyzer] Recipe file written to: {output_path}")
    logging.info("[SqnrAnalyzer] Generated classes:")
    for cls in generated_classes:
        logging.info(f"  - {cls}")
    logging.info(
        "[SqnrAnalyzer] Replace the original recipe class in your export script "
        "with one of the above and re-run calibration to evaluate accuracy."
    )
