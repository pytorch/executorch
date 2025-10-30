import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import read_excel_with_json_header
from tabulate import tabulate


def print_section_header(title):
    """Print a clearly visible section header to stdout"""
    print("\n\n" + "=" * 100)
    print(f"===== {title} ".ljust(99, "="))
    print("=" * 100 + "\n")


def normalize_name(name):
    """Normalize tab name for better matching"""
    # Convert to lowercase and remove spaces
    return name.lower().replace(" ", "").replace("(private)", "")


def parse_model_device_config(config):
    """Extract model and device from config"""
    model = config.get("model", "")
    backend = config.get("backend", "")
    full_model = f"{model}({backend})" if backend else model
    base_device = config.get("device", "")
    os_version = config.get("arch", "")
    full_device = f"{base_device}({os_version})" if os_version else base_device
    if not base_device:
        return full_model, "unkown", "unknown", ""
    return full_model, full_device, base_device, os_version


def is_matching_dataset(primary_sheet, reference_sheet):
    """
    Check if two datasets match for comparison based on model and device
    Allows different OS versions for the same device
    """
    primary_model = normalize_name(primary_sheet.get("model", ""))
    primary_device = normalize_name(primary_sheet.get("base_device", ""))
    # primary_os = normalize_name(primary_sheet.get("os_version", ""))

    reference_model = normalize_name(reference_sheet.get("model", ""))
    reference_device = normalize_name(reference_sheet.get("base_device", ""))
    # reference_os = normalize_name(reference_sheet.get("os_version", ""))

    if not primary_model:
        print("Warning: Primary sheet {} has no model info, for {primary_model} ")
        return False
    if not reference_model:
        print("Warning: Reference sheet {} has no model info, for {reference_model}")
        return False

    # Model must match exactly
    if primary_model != reference_model:
        return False

    # Device base name must match exactly
    if primary_device != reference_device:
        return False

    return True


def analyze_latency_stability(  # noqa: C901
    target_metric,
    primary_file,
    reference_file=None,
    output_dir="stability_analysis_results",
    verbose_level=0,
):
    print_section_header(f"Analyzing Stability Against Metric '{target_metric}'")
    print(f"Primary dataset: {primary_file}")
    if reference_file:
        print(f"Reference dataset for comparison: {reference_file}")

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load primary datasets
    print_section_header("LOADING PRIMARY DATASETS (Private)")
    primary_datasets = {}
    documents = read_excel_with_json_header(primary_file)

    if verbose_level > 2:
        print(f"Printing documents: {documents}")

    for document in documents:
        sheetName = document.get("sheetName", None)
        df = document.get("df", None)
        config = document.get("groupInfo", None)
        print(f"Loading dataset: {sheetName} with config: {config} ")

        if df is None or df.empty:
            print(f"Skipping sheet {sheetName} because it has no df data")
            continue

        if not config or not sheetName:
            print(
                f" Skipping document: Missing required data groupInfo:{config} sheetName:{sheetName}"
            )
            continue

        model, full_device, base_device, os_version = parse_model_device_config(config)

        # Skip sheets without required columns
        required_cols = [target_metric, "metadata_info.timestamp"]
        if not all(col in df.columns for col in required_cols):
            print(f"  Skipping {sheetName}: Missing required columns")
            continue

        # Convert Date to datetime
        df["Date"] = pd.to_datetime(df["metadata_info.timestamp"])

        # Calculate stability metrics along the target column in the dataset
        metrics = calculate_stability_metrics(
            df,
            target_metric,
        )

        primary_datasets[sheetName] = {
            "df": df,
            "metrics": metrics,
            "model": model,
            "full_device": full_device,
            "base_device": base_device,
            "os_version": os_version,
            "sheet_name": sheetName,
        }

    # Load reference datasets if provided
    reference_datasets = {}
    if reference_file:
        print_section_header("LOADING REFERENCE DATASETS (Public)")
        documents = read_excel_with_json_header(reference_file)

        for document in documents:
            sheetName = document.get("sheetName", None)
            df = document.get("df", None)
            config = document.get("groupInfo", None)
            print(f"Loading dataset: {sheetName} with config:{config}")
            if df is None or df.empty:
                print(f"Skipping sheet {sheetName} because it has no df data")
                continue

            if not config or not sheetName:
                print(
                    f" Skipping document: Missing required data groupInfo:{config} sheetName:{sheetName}"
                )
                continue

            model, full_device, base_device, os_version = parse_model_device_config(
                config
            )

            # Skip sheets without required columns
            required_cols = [target_metric, "metadata_info.timestamp"]
            if not all(col in df.columns for col in required_cols):
                print(
                    f"  Skipping reference {sheetName}: Missing required columns{required_cols}"
                )
                continue

            # Convert Date to datetime
            df["Date"] = pd.to_datetime(df["metadata_info.timestamp"])

            # Calculate stability metrics
            metrics = calculate_stability_metrics(
                df,
                target_metric,
            )

            reference_datasets[sheetName] = {
                "df": df,
                "metrics": metrics,
                "model": model,
                "full_device": full_device,
                "sheet_name": sheetName,
                "base_device": base_device,
                "os_version": os_version,
            }

    # Process primary datasets
    if verbose_level > 2:
        print_section_header("ANALYZING PRIMARY DATASETS")
        for sheet, info in primary_datasets.items():
            # Generate dataset report
            generate_dataset_report(
                sheet,
                target_metric,
                info["model"],
                info["full_device"],
                "Primary",
                info["df"],
                info["metrics"],
                output_dir,
            )

            # Generate time series plot
            if len(info["df"]) > 5:  # Only create plot if enough data points
                generate_time_series_plot(sheet, info["df"], output_dir, "Primary")

    # Process reference datasets if provided
    if reference_file and verbose_level > 3:
        print_section_header("ANALYZING REFERENCE DATASETS")
        for sheet, info in reference_datasets.items():
            # Generate dataset report
            generate_dataset_report(
                sheet,
                target_metric,
                info["model"],
                info["full_device"],
                "Reference",
                info["df"],
                info["metrics"],
                output_dir,
            )

            # Generate time series plot
            if len(info["df"]) > 5:  # Only create plot if enough data points
                generate_time_series_plot(sheet, info["df"], output_dir, "Reference")

    # Generate comparison reports for matching datasets
    if reference_file and verbose_level > 1:
        print_section_header("PRIVATE VS PUBLIC STABILITY COMPARISON")
        matches_found = False

        for primary_sheet, primary_info in primary_datasets.items():
            found_match = False

            for ref_sheet, ref_info in reference_datasets.items():
                if is_matching_dataset(primary_info, ref_info):
                    # Found a match
                    print(
                        f"Matched: {primary_sheet} (Private) with {ref_sheet} (Public)"
                    )
                    generate_comparison_report(
                        primary_sheet,
                        ref_sheet,
                        primary_info,
                        ref_info,
                        output_dir,
                    )
                    found_match = True
                    matches_found = True
                    break

            if not found_match:
                print(
                    f"Warning: No matching reference dataset for {primary_sheet} with config:  {primary_info['model']}{primary_info['full_device']} "
                )

        if not matches_found:
            print("No matching datasets found between primary and reference files.")

    if verbose_level > 0:
        # Generate intra-primary summary (comparing across different models/devices)
        print_section_header("INTRA-PRIMARY STABILITY COMPARISON")
        generate_intra_primary_summary(primary_datasets, output_dir)

    # Generate summary report for all datasets
    print_section_header("COMPREHENSIVE STABILITY SUMMARY")
    generate_summary_report(
        primary_datasets, reference_datasets if reference_file else None, output_dir
    )

    print(f"\nAnalysis complete. Results saved to {output_dir}/")
    return primary_datasets, reference_datasets if reference_file else None


def calculate_stability_metrics(  # noqa: C901
    df,
    target_metric,
):
    """Calculate stability metrics for the given dataset"""
    metrics = {}
    # Extract data and ingore NaN values
    raw_latency = df[target_metric].dropna().values

    # Central tendency metrics
    metrics["mean_raw_latency"] = np.mean(raw_latency)
    metrics["median_raw_latency"] = np.median(raw_latency)

    # Dispersion metrics
    metrics["std_raw_latency"] = np.std(raw_latency, ddof=1)
    metrics["cv_raw_latency"] = (
        metrics["std_raw_latency"] / metrics["mean_raw_latency"]
    ) * 100
    metrics["iqr_raw_latency"] = np.percentile(raw_latency, 75) - np.percentile(
        raw_latency, 25
    )

    # Percentile metrics
    for p in [50, 90, 95, 99]:
        metrics[f"p{p}_raw_latency"] = np.percentile(raw_latency, p)

    # Inter-jitter metrics (variability between runs)
    if np.min(raw_latency) > 0:
        metrics["max_min_range_ratio_raw"] = np.max(raw_latency) / np.min(raw_latency)
    else:
        metrics["max_min_range_ratio_raw"] = float("inf")
        print("Warning: Minimum latency value is zero, max/min ratio set to infinity")

    metrics["p99_p50_ratio_raw"] = (
        metrics["p99_raw_latency"] / metrics["p50_raw_latency"]
    )

    # Intra-jitter proxy (if both raw and trimmed latency are available)
    trimmed_metric_col = "trimmean_inference_latency(ms)"
    if (
        target_metric == "avg_inference_latency(ms)"
        and trimmed_metric_col in df.columns
    ):
        trimmed_latency = df[trimmed_metric_col].values
        if trimmed_latency is not None:
            metrics["mean_trimmed_latency"] = np.mean(trimmed_latency)
            metrics["median_trimmed_latency"] = np.median(trimmed_latency)
            metrics["std_trimmed_latency"] = np.std(trimmed_latency, ddof=1)
            metrics["cv_trimmed_latency"] = (
                metrics["std_trimmed_latency"] / metrics["mean_trimmed_latency"]
            ) * 100
            metrics["iqr_trimmed_latency"] = np.percentile(
                trimmed_latency, 75
            ) - np.percentile(trimmed_latency, 25)
            for p in [50, 90, 95, 99]:
                metrics[f"p{p}_trimmed_latency"] = np.percentile(trimmed_latency, p)
            if np.min(trimmed_latency) > 0:
                metrics["max_min_range_ratio_trimmed"] = np.max(
                    trimmed_latency
                ) / np.min(trimmed_latency)
            else:
                metrics["max_min_range_ratio_trimmed"] = float("inf")
                print(
                    "Warning: Minimum trimmed latency value is zero, max/min ratio set to infinity"
                )
            metrics["p99_p50_ratio_trimmed"] = (
                metrics["p99_trimmed_latency"] / metrics["p50_trimmed_latency"]
            )
            trimming_effect = (raw_latency - trimmed_latency) / raw_latency
            metrics["mean_trimming_effect_ratio"] = np.mean(trimming_effect)
            metrics["max_trimming_effect_ratio"] = np.max(trimming_effect)

    # Time-based stability (rolling window of 5 samples)
    if len(df) >= 5:
        df_sorted = df.sort_values("Date")
        rolling_std = df_sorted[target_metric].rolling(window=5).std()
        metrics["mean_rolling_std"] = rolling_std.mean()
        metrics["max_rolling_std"] = rolling_std.max()

    # Stability score calculation (0-100 scale)
    # Weights for different components
    cv_weight = 0.5
    max_min_weight = 0.25
    p99_p50_weight = 0.25

    # Convert metrics to scores (lower is better for all these metrics)
    cv_score = max(
        0, 100 - (metrics["cv_raw_latency"] * 10)
    )  # CV of 10% or more gets 0

    if metrics["max_min_range_ratio_raw"] == float("inf"):
        max_min_score = 0
    else:
        max_min_score = max(
            0, 100 - ((metrics["max_min_range_ratio_raw"] - 1) * 50)
        )  # Ratio of 3.0 or more gets 0

    p99_p50_score = max(
        0, 100 - ((metrics["p99_p50_ratio_raw"] - 1) * 100)
    )  # Ratio of 2.0 or more gets 0

    # Weighted average
    metrics["stability_score"] = (
        cv_weight * cv_score
        + max_min_weight * max_min_score
        + p99_p50_weight * p99_p50_score
    )

    # Stability rating based on score
    if metrics["stability_score"] >= 90:
        metrics["stability_rating"] = "Excellent"
    elif metrics["stability_score"] >= 80:
        metrics["stability_rating"] = "Good"
    elif metrics["stability_score"] >= 60:
        metrics["stability_rating"] = "Moderate"
    else:
        metrics["stability_rating"] = "Poor"

    return metrics


def generate_dataset_report(  # noqa: C901
    sheet_name, target_column, model, device, dataset_type, df, metrics, output_dir
):
    """Generate a detailed report for a single dataset"""
    report_file = f"{output_dir}/{sheet_name}_{dataset_type.lower()}_report.txt"

    # Create a string buffer to hold the report content
    report_content = []

    # Header
    report_content.append(f"Latency Stability Analysis: {sheet_name} ({dataset_type})")
    report_content.append("=" * 80)
    report_content.append(f"Model: {model}")
    report_content.append(f"Device: {device}")
    report_content.append("")

    # Dataset overview
    report_content.append("Dataset Overview:")
    report_content.append(
        f"  - Number of samples: {len(df[target_column].dropna().values)}"
    )
    report_content.append(f"  - Date range: {df['Date'].min()} to {df['Date'].max()}")
    report_content.append("")

    # Central tendency metrics
    report_content.append("Central Tendency Metrics:")
    report_content.append(f"  - Mean latency: {metrics['mean_raw_latency']:.2f} ms")
    report_content.append(
        f"  - Median latency (P50): {metrics['median_raw_latency']:.2f} ms"
    )
    if (
        "mean_trimmed_latency" in metrics
        and metrics["mean_trimmed_latency"] is not None
    ):
        report_content.append(
            f"  - Mean trimmed latency: {metrics['mean_trimmed_latency']:.2f} ms"
        )
        report_content.append(
            f"  - Median trimmed latency: {metrics['median_trimmed_latency']:.2f} ms"
        )
    report_content.append("")

    # Dispersion metrics
    report_content.append("Dispersion Metrics:")
    report_content.append(
        f"  - Standard deviation: {metrics['std_raw_latency']:.2f} ms"
    )
    report_content.append(
        f"  - Coefficient of variation (CV): {metrics['cv_raw_latency']:.2f}%"
    )
    report_content.append(
        f"  - Interquartile range (IQR): {metrics['iqr_raw_latency']:.2f} ms"
    )
    if "std_trimmed_latency" in metrics and metrics["std_trimmed_latency"] is not None:
        report_content.append(
            f"  - Trimmed standard deviation: {metrics['std_trimmed_latency']:.2f} ms"
        )
        report_content.append(
            f"  - Trimmed coefficient of variation: {metrics['cv_trimmed_latency']:.2f}%"
        )
    report_content.append("")

    # Percentile metrics
    report_content.append("Percentile Metrics:")
    report_content.append(f"  - P50 (median): {metrics['p50_raw_latency']:.2f} ms")
    report_content.append(f"  - P90: {metrics['p90_raw_latency']:.2f} ms")
    report_content.append(f"  - P95: {metrics['p95_raw_latency']:.2f} ms")
    report_content.append(f"  - P99: {metrics['p99_raw_latency']:.2f} ms")
    report_content.append("")

    # Jitter metrics
    report_content.append("Inter-Jitter Metrics (variability between runs):")
    if metrics["max_min_range_ratio_raw"] == float("inf"):
        report_content.append("  - Max/Min ratio: Infinity (minimum value is zero)")
    else:
        report_content.append(
            f"  - Max/Min ratio: {metrics['max_min_range_ratio_raw']:.4f}"
        )
    report_content.append(f"  - P99/P50 ratio: {metrics['p99_p50_ratio_raw']:.4f}")
    if "mean_rolling_std" in metrics:
        report_content.append(
            f"  - Mean rolling std (window=5): {metrics['mean_rolling_std']:.2f} ms"
        )
    report_content.append("")

    if (
        "mean_trimming_effect_ratio" in metrics
        and metrics["mean_trimming_effect_ratio"] is not None
    ):
        report_content.append("Intra-Jitter Metrics (variability within runs):")
        report_content.append(
            f"  - Mean trimming effect ratio: {metrics['mean_trimming_effect_ratio']*100:.2f}%"
        )
        report_content.append(
            f"  - Max trimming effect ratio: {metrics['max_trimming_effect_ratio']*100:.2f}%"
        )
        report_content.append("")

    # TPS metrics
    if "mean_tps" in metrics and metrics["mean_tps"] is not None:
        report_content.append("Throughput Metrics:")
        report_content.append(f"  - Mean TPS: {metrics['mean_tps']:.2f}")
        report_content.append(
            f"  - TPS coefficient of variation: {metrics['cv_tps']:.2f}%"
        )
        report_content.append("")

    # Stability assessment
    report_content.append("Stability Assessment:")
    report_content.append(
        f"  - Overall stability score: {metrics['stability_score']:.1f}/100"
    )
    report_content.append(
        f"  - Overall stability rating: {metrics['stability_rating']}"
    )
    report_content.append("")

    # Interpretation
    report_content.append("Interpretation:")

    # Stability rating explanation
    if metrics["stability_rating"] == "Excellent":
        report_content.append(
            f"  The benchmark shows excellent stability (score: {metrics['stability_score']:.1f}/100) with very low"
        )
        report_content.append(
            f"  variation between runs (CV: {metrics['cv_raw_latency']:.2f}%)."
        )
        report_content.append(
            "  This indicates highly consistent performance suitable for latency-sensitive applications."
        )
    elif metrics["stability_rating"] == "Good":
        report_content.append(
            f"  The benchmark shows good stability (score: {metrics['stability_score']:.1f}/100) with low"
        )
        report_content.append(
            f"  variation between runs (CV: {metrics['cv_raw_latency']:.2f}%)."
        )
        report_content.append(
            "  Performance is consistent and predictable for most use cases."
        )
    elif metrics["stability_rating"] == "Moderate":
        report_content.append(
            f"  The benchmark shows moderate stability (score: {metrics['stability_score']:.1f}/100) with noticeable"
        )
        report_content.append(
            f"  variation between runs (CV: {metrics['cv_raw_latency']:.2f}%)."
        )
        report_content.append(
            "  While average performance is acceptable, occasional latency spikes may occur."
        )
    else:
        report_content.append(
            f"  The benchmark shows poor stability (score: {metrics['stability_score']:.1f}/100) with significant"
        )
        report_content.append(
            f"  variation between runs (CV: {metrics['cv_raw_latency']:.2f}%)."
        )
        report_content.append(
            "  Performance is unpredictable and may lead to inconsistent user experience."
        )

    # Additional insights
    if (
        "mean_trimming_effect_ratio" in metrics
        and metrics["mean_trimming_effect_ratio"] is not None
        and metrics["mean_trimming_effect_ratio"] > 0.05
    ):
        report_content.append("")
        report_content.append(
            "  The significant difference between raw and trimmed means suggests"
        )
        report_content.append(
            f"  considerable intra-run jitter ({metrics['mean_trimming_effect_ratio']*100:.1f}%) with occasional outliers within benchmark runs."
        )

    if (
        metrics["max_min_range_ratio_raw"] != float("inf")
        and metrics["max_min_range_ratio_raw"] > 1.2
    ):
        report_content.append("")
        report_content.append(
            f"  The max/min ratio of {metrics['max_min_range_ratio_raw']:.2f} indicates"
        )
        report_content.append(
            "  substantial performance differences between the best and worst runs."
        )

    if metrics["p99_p50_ratio_raw"] > 1.1:
        report_content.append("")
        report_content.append(
            f"  The P99/P50 ratio of {metrics['p99_p50_ratio_raw']:.2f} suggests"
        )
        report_content.append(
            "  occasional latency spikes that could affect tail latency sensitive applications."
        )

    # Join all content with newlines to create the full report
    full_report = "\n".join(report_content)

    # Write to file
    with open(report_file, "w") as f:
        f.write(full_report)

    # Also print to stdout
    print("\n" + full_report + "\n")
    print("=" * 80)


def generate_time_series_plot(dataset_name, df, output_dir, dataset_type):
    """Generate time series plot of latency values"""
    plt.figure(figsize=(12, 6))

    # Sort by date
    df_sorted = df.sort_values("Date")

    # Plot raw latency
    plt.plot(
        df_sorted["Date"],
        df_sorted["avg_inference_latency(ms)"],
        "b-",
        label="Raw Latency",
    )

    # Plot trimmed latency if available
    if "trimmean_inference_latency(ms)" in df_sorted.columns:
        plt.plot(
            df_sorted["Date"],
            df_sorted["trimmean_inference_latency(ms)"],
            "g-",
            label="Trimmed Latency",
        )

    # Add rolling mean
    window = min(5, len(df_sorted))
    if window > 1:
        rolling_mean = (
            df_sorted["avg_inference_latency(ms)"].rolling(window=window).mean()
        )
        plt.plot(
            df_sorted["Date"], rolling_mean, "r--", label=f"{window}-point Rolling Mean"
        )

    plt.title(f"Latency Over Time: {dataset_name} ({dataset_type})")
    plt.xlabel("Date")
    plt.ylabel("Latency (ms)")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot
    plt.savefig(f"{output_dir}/{dataset_name}_{dataset_type.lower()}_time_series.png")
    print(
        f"Generated time series plot: {output_dir}/{dataset_name}_{dataset_type.lower()}_time_series.png"
    )
    plt.close()


def generate_comparison_report(  # noqa: C901
    primary_sheet,
    reference_sheet,
    primary_info,
    reference_info,
    output_dir,
):
    """Generate a comparison report between primary and reference datasets"""
    report_file = f"{output_dir}/{primary_sheet}_vs_{reference_sheet}_comparison.txt"

    # Create a string buffer to hold the report content
    report_content = []

    model = (primary_info["model"],)
    primary_device = (primary_info["full_device"],)
    reference_device = reference_info["full_device"]
    primary_metrics = primary_info["metrics"]
    reference_metrics = reference_info["metrics"]

    # Header
    report_content.append("Private vs Public Stability Comparison")
    report_content.append("=" * 80)
    report_content.append(f"Private Dataset: {primary_sheet}")
    report_content.append(f"Public Dataset: {reference_sheet}")
    report_content.append(f"Model: {model}")
    report_content.append(f"Private Device: {primary_device}")
    report_content.append(f"Public Device: {reference_device}")
    report_content.append("")

    # Create comparison table
    report_content.append("Metric Comparison:")

    # Format the metrics table
    headers = [
        "Metric",
        "Private (Primary)",
        "Public (Reference)",
        "Difference",
        "% Change",
    ]
    rows = []

    # Add key metrics to the table
    metrics_to_compare = [
        ("Mean Value", "mean_raw_latency", ""),
        ("Median Value", "median_raw_latency", ""),
        ("Standard Deviation", "std_raw_latency", ""),
        ("CV (%)", "cv_raw_latency", "%"),
        ("IQR", "iqr_raw_latency", ""),
        ("P99", "p99_raw_latency", ""),
        ("Max/Min Ratio", "max_min_range_ratio_raw", ""),
        ("P99/P50 Ratio", "p99_p50_ratio_raw", ""),
        ("Stability Score", "stability_score", ""),
    ]

    for label, key, unit in metrics_to_compare:
        if key in primary_metrics and key in reference_metrics:
            primary_val = primary_metrics[key]
            reference_val = reference_metrics[key]

            # Handle infinity values
            if primary_val == float("inf") or reference_val == float("inf"):
                if primary_val == float("inf") and reference_val == float("inf"):
                    diff = 0
                    pct_change = 0
                elif primary_val == float("inf"):
                    diff = float("inf")
                    pct_change = float("inf")
                else:
                    diff = float("-inf")
                    pct_change = -100
            else:
                diff = primary_val - reference_val
                # Calculate percent change, avoiding division by zero
                if reference_val != 0:
                    pct_change = (diff / reference_val) * 100
                else:
                    pct_change = float("inf")

            # Format values based on the metric
            if key == "stability_score":
                if primary_val == float("inf"):
                    primary_str = "Infinity"
                else:
                    primary_str = f"{primary_val:.1f}/100"

                if reference_val == float("inf"):
                    reference_str = "Infinity"
                else:
                    reference_str = f"{reference_val:.1f}/100"

                if diff == float("inf"):
                    diff_str = "Infinity"
                elif diff == float("-inf"):
                    diff_str = "-Infinity"
                else:
                    diff_str = f"{diff:.1f}"

                if pct_change == float("inf"):
                    pct_str = "Infinity"
                elif pct_change == float("-inf"):
                    pct_str = "-Infinity"
                else:
                    pct_str = f"{pct_change:.1f}%"

                row = [label, primary_str, reference_str, diff_str, pct_str]
            elif unit == "%":
                if primary_val == float("inf"):
                    primary_str = "Infinity%"
                else:
                    primary_str = f"{primary_val:.2f}%"

                if reference_val == float("inf"):
                    reference_str = "Infinity%"
                else:
                    reference_str = f"{reference_val:.2f}%"

                if diff == float("inf"):
                    diff_str = "Infinity%"
                elif diff == float("-inf"):
                    diff_str = "-Infinity%"
                else:
                    diff_str = f"{diff:.2f}%"

                if pct_change == float("inf"):
                    pct_str = "Infinity%"
                elif pct_change == float("-inf"):
                    pct_str = "-Infinity%"
                else:
                    pct_str = f"{pct_change:.1f}%"

                row = [label, primary_str, reference_str, diff_str, pct_str]
            elif unit == "ms":
                if primary_val == float("inf"):
                    primary_str = "Infinity ms"
                else:
                    primary_str = f"{primary_val:.2f} ms"

                if reference_val == float("inf"):
                    reference_str = "Infinity ms"
                else:
                    reference_str = f"{reference_val:.2f} ms"

                if diff == float("inf"):
                    diff_str = "Infinity ms"
                elif diff == float("-inf"):
                    diff_str = "-Infinity ms"
                else:
                    diff_str = f"{diff:.2f} ms"

                if pct_change == float("inf"):
                    pct_str = "Infinity%"
                elif pct_change == float("-inf"):
                    pct_str = "-Infinity%"
                else:
                    pct_str = f"{pct_change:.1f}%"

                row = [label, primary_str, reference_str, diff_str, pct_str]
            else:
                if primary_val == float("inf"):
                    primary_str = "Infinity"
                else:
                    primary_str = f"{primary_val:.4f}"

                if reference_val == float("inf"):
                    reference_str = "Infinity"
                else:
                    reference_str = f"{reference_val:.4f}"

                if diff == float("inf"):
                    diff_str = "Infinity"
                elif diff == float("-inf"):
                    diff_str = "-Infinity"
                else:
                    diff_str = f"{diff:.4f}"

                if pct_change == float("inf"):
                    pct_str = "Infinity%"
                elif pct_change == float("-inf"):
                    pct_str = "-Infinity%"
                else:
                    pct_str = f"{pct_change:.1f}%"

                row = [label, primary_str, reference_str, diff_str, pct_str]

            rows.append(row)

    # Add stability ratings
    rows.append(
        [
            "Stability Rating",
            primary_metrics["stability_rating"],
            reference_metrics["stability_rating"],
            "N/A",
            "N/A",
        ]
    )

    # Format the table
    table = tabulate(rows, headers=headers, tablefmt="grid")
    report_content.append(table)
    report_content.append("")

    # Add interpretation
    report_content.append("Interpretation:")

    # Compare stability scores
    if primary_metrics["stability_score"] > reference_metrics["stability_score"]:
        if reference_metrics["stability_score"] != 0:
            diff_pct = (
                (
                    primary_metrics["stability_score"]
                    - reference_metrics["stability_score"]
                )
                / reference_metrics["stability_score"]
                * 100
            )
            report_content.append(
                f"  Private environment shows better stability with a {diff_pct:.1f}% higher stability score."
            )
        else:
            report_content.append("  Private environment shows better stability.")
        report_content.append(
            f"  (Private: {primary_metrics['stability_score']:.1f}/100 vs Public: {reference_metrics['stability_score']:.1f}/100)"
        )
    elif primary_metrics["stability_score"] < reference_metrics["stability_score"]:
        if primary_metrics["stability_score"] != 0:
            diff_pct = (
                (
                    reference_metrics["stability_score"]
                    - primary_metrics["stability_score"]
                )
                / reference_metrics["stability_score"]
                * 100
            )
            report_content.append(
                f"  Public environment shows better stability with a {diff_pct:.1f}% higher stability score."
            )
        else:
            report_content.append("  Public environment shows better stability.")
        report_content.append(
            f"  (Private: {primary_metrics['stability_score']:.1f}/100 vs Public: {reference_metrics['stability_score']:.1f}/100)"
        )
    else:
        report_content.append("  Both environments show identical stability scores.")

    # Compare CV values
    if primary_metrics["cv_raw_latency"] < reference_metrics["cv_raw_latency"]:
        if reference_metrics["cv_raw_latency"] != 0:
            diff_pct = (
                (
                    reference_metrics["cv_raw_latency"]
                    - primary_metrics["cv_raw_latency"]
                )
                / reference_metrics["cv_raw_latency"]
                * 100
            )
            report_content.append(
                f"  Private environment has {diff_pct:.1f}% lower coefficient of variation, indicating more consistent performance."
            )
        else:
            report_content.append(
                "  Private environment has lower coefficient of variation, indicating more consistent performance."
            )
    elif primary_metrics["cv_raw_latency"] > reference_metrics["cv_raw_latency"]:
        if reference_metrics["cv_raw_latency"] != 0:
            diff_pct = (
                (
                    primary_metrics["cv_raw_latency"]
                    - reference_metrics["cv_raw_latency"]
                )
                / reference_metrics["cv_raw_latency"]
                * 100
            )
            report_content.append(
                f"  Public environment has {diff_pct:.1f}% lower coefficient of variation, indicating more consistent performance."
            )
        else:
            report_content.append(
                "  Public environment has lower coefficient of variation, indicating more consistent performance."
            )

    # Compare latency
    if primary_metrics["mean_raw_latency"] < reference_metrics["mean_raw_latency"]:
        if reference_metrics["mean_raw_latency"] != 0:
            diff_pct = (
                (
                    reference_metrics["mean_raw_latency"]
                    - primary_metrics["mean_raw_latency"]
                )
                / reference_metrics["mean_raw_latency"]
                * 100
            )
            report_content.append(
                f"  Private environment has {diff_pct:.1f}% lower mean latency, indicating better performance."
            )
        else:
            report_content.append(
                "  Private environment has lower mean latency, indicating better performance."
            )
    elif primary_metrics["mean_raw_latency"] > reference_metrics["mean_raw_latency"]:
        if primary_metrics["mean_raw_latency"] != 0:
            diff_pct = (
                (
                    primary_metrics["mean_raw_latency"]
                    - reference_metrics["mean_raw_latency"]
                )
                / reference_metrics["mean_raw_latency"]
                * 100
            )
            report_content.append(
                f"  Public environment has {diff_pct:.1f}% lower mean latency, indicating better performance."
            )
        else:
            report_content.append(
                "  Public environment has lower mean latency, indicating better performance."
            )

    # Note about OS version difference if applicable
    primary_device_base = primary_info.get("base_device", "")
    primary_os = primary_info.get("os_version", "")
    reference_device_base = reference_info.get("base_device", "")
    reference_os = reference_info.get("os_version", "")

    if primary_os != reference_os and primary_os and reference_os:
        report_content.append("")
        report_content.append(
            f"  Note: This comparison is between {primary_device_base} with {primary_os} (Private) and"
        )
        report_content.append(
            f"  {reference_device_base} with {reference_os} (Public). OS version differences may"
        )
        report_content.append("  contribute to observed stability variations.")

    # Recommendation
    report_content.append("")
    report_content.append("Recommendation:")
    if primary_metrics["stability_score"] > reference_metrics["stability_score"]:
        report_content.append(
            "  The private environment provides better stability for this model+device combination."
        )
        report_content.append(
            "  It is recommended for applications where consistent performance is critical."
        )
    elif primary_metrics["stability_score"] < reference_metrics["stability_score"]:
        report_content.append(
            "  The public environment provides better stability for this model+device combination."
        )
        report_content.append(
            "  Consider investigating factors affecting stability in the private environment."
        )
    else:
        report_content.append(
            "  Both environments provide similar stability. Other factors like cost or availability"
        )
        report_content.append("  may be considered for choosing between them.")

    # Join all content with newlines to create the full report
    full_report = "\n".join(report_content)

    # Write to file
    with open(report_file, "w") as f:
        f.write(full_report)

    # Also print to stdout
    print("\n" + full_report + "\n")
    print("=" * 80)


def generate_intra_primary_summary(primary_datasets, output_dir):  # noqa: C901
    """Generate a summary comparing different models and devices within the primary dataset"""
    report_file = f"{output_dir}/intra_primary_stability_summary.txt"

    # Extract relevant data for comparison
    data = []
    for sheet_name, info in primary_datasets.items():
        data.append(
            {
                "Sheet": sheet_name,
                "Model": info["model"],
                "Device": info["full_device"],
                "Mean Value": info["metrics"]["mean_raw_latency"],
                "CV (%)": info["metrics"]["cv_raw_latency"],
                "Stability Score": info["metrics"]["stability_score"],
                "Stability Rating": info["metrics"]["stability_rating"],
                "Max/Min Ratio": info["metrics"]["max_min_range_ratio_raw"],
                "P99/P50 Ratio": info["metrics"]["p99_p50_ratio_raw"],
            }
        )

    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(data)

    # Sort by stability score (descending)
    df = df.sort_values("Stability Score", ascending=False)

    # Create a string buffer to hold the report content
    report_content = []

    # Header
    report_content.append("Intra-Primary Stability Comparison")
    report_content.append("=" * 80)
    report_content.append("")

    # Overall summary table
    report_content.append("Overall Summary:")
    report_content.append(
        tabulate(df, headers="keys", tablefmt="grid", floatfmt=".2f", showindex=False)
    )
    report_content.append("")

    # Best and worst performers
    best_dataset = df.loc[df["Stability Score"].idxmax()]
    worst_dataset = df.loc[df["Stability Score"].idxmin()]

    report_content.append("Best and Worst Performers:")
    report_content.append(
        f"  Best stability: {best_dataset['Sheet']} (Score: {best_dataset['Stability Score']:.1f}/100)"
    )
    report_content.append(
        f"  Worst stability: {worst_dataset['Sheet']} (Score: {worst_dataset['Stability Score']:.1f}/100)"
    )
    report_content.append("")

    # Model-based comparison if multiple models exist
    models = df["Model"].unique()
    if len(models) > 1:
        report_content.append("Model-based Comparison:")
        model_stats = df.groupby("Model").agg(
            {
                "Stability Score": ["mean", "min", "max"],
                "CV (%)": ["mean", "min", "max"],
            }
        )

        # Sort by mean stability score (descending)
        model_stats = model_stats.sort_values(
            ("Stability Score", "mean"), ascending=False
        )

        report_content.append(
            tabulate(model_stats, headers="keys", tablefmt="grid", floatfmt=".2f")
        )

        best_model = model_stats["Stability Score"]["mean"].idxmax()
        report_content.append(
            f"  Most stable model: {best_model} (Avg. Score: {model_stats.loc[best_model, ('Stability Score', 'mean')]:.1f}/100)"
        )
        report_content.append("")

    # Device-based comparison
    # First, extract base device names for grouping
    device_base_map = {}
    for sheet_name, info in primary_datasets.items():
        device_base = info.get("base_device", "")
        device_base_map[sheet_name] = device_base

    # Add base device to DataFrame
    df["Device Base"] = df["Sheet"].map(device_base_map)

    # Group by base device
    device_bases = df["Device Base"].unique()
    if len(device_bases) > 1:
        report_content.append("Device-based Comparison (Grouped by Base Device):")
        device_stats = df.groupby("Device Base").agg(
            {
                "Stability Score": ["mean", "min", "max"],
                "CV (%)": ["mean", "min", "max"],
            }
        )

        # Sort by mean stability score (descending)
        device_stats = device_stats.sort_values(
            ("Stability Score", "mean"), ascending=False
        )

        report_content.append(
            tabulate(device_stats, headers="keys", tablefmt="grid", floatfmt=".2f")
        )

        best_device = device_stats["Stability Score"]["mean"].idxmax()
        report_content.append(
            f"  Most stable device: {best_device} (Avg. Score: {device_stats.loc[best_device, ('Stability Score', 'mean')]:.1f}/100)"
        )
        report_content.append("")

    # OS version comparison if multiple OS versions exist
    os_versions = {}
    for sheet_name, info in primary_datasets.items():
        os_version = info.get("os_version", "")
        if os_version:  # Only include if OS version was extracted
            os_versions[sheet_name] = os_version

    if os_versions and len(set(os_versions.values())) > 1:
        # Add OS version to DataFrame
        df["OS Version"] = df["Sheet"].map(os_versions)

        # Remove rows with no OS version
        df_os = df[df["OS Version"].notna()]

        if len(df_os) > 0:
            report_content.append("OS Version Comparison:")
            os_stats = df_os.groupby("OS Version").agg(
                {
                    "Stability Score": ["mean", "min", "max"],
                    "CV (%)": ["mean", "min", "max"],
                }
            )

            # Sort by mean stability score (descending)
            os_stats = os_stats.sort_values(
                ("Stability Score", "mean"), ascending=False
            )

            report_content.append(
                tabulate(os_stats, headers="keys", tablefmt="grid", floatfmt=".2f")
            )

            best_os = os_stats["Stability Score"]["mean"].idxmax()
            report_content.append(
                f"  Most stable OS version: {best_os} (Avg. Score: {os_stats.loc[best_os, ('Stability Score', 'mean')]:.1f}/100)"
            )
            report_content.append("")

    # Insights and recommendations
    report_content.append("Insights and Recommendations:")

    # Check for patterns in stability
    if len(models) > 1:
        model_cv = df.groupby("Model")["CV (%)"].mean()
        most_stable_model = model_cv.idxmin()
        least_stable_model = model_cv.idxmax()
        report_content.append(
            f"  - {most_stable_model} shows the most consistent performance across devices."
        )
        report_content.append(
            f"  - {least_stable_model} shows more variability and may need further optimization."
        )

    if len(device_bases) > 1:
        device_cv = df.groupby("Device Base")["CV (%)"].mean()
        most_stable_device = device_cv.idxmin()
        least_stable_device = device_cv.idxmax()
        report_content.append(
            f"  - {most_stable_device} provides the most stable environment for model execution."
        )
        report_content.append(
            f"  - {least_stable_device} shows higher variability and may not be ideal for latency-sensitive applications."
        )

    if os_versions and len(set(os_versions.values())) > 1 and len(df_os) > 0:
        os_cv = df_os.groupby("OS Version")["CV (%)"].mean()
        most_stable_os = os_cv.idxmin()
        least_stable_os = os_cv.idxmax()
        report_content.append(
            f"  - {most_stable_os} provides better stability than {least_stable_os} across tested devices."
        )

    # General recommendations
    report_content.append(
        "  - For critical applications requiring consistent performance, prefer:"
    )
    if len(models) > 1:
        report_content.append(f"    * Model: {best_model}")
    else:
        report_content.append(f"    * Model: {df['Model'].iloc[0]}")

    if len(device_bases) > 1:
        report_content.append(f"    * Device: {best_device}")
    else:
        report_content.append(f"    * Device: {df['Device Base'].iloc[0]}")

    if os_versions and len(set(os_versions.values())) > 1 and len(df_os) > 0:
        report_content.append(f"    * OS Version: {best_os}")

    # Join all content with newlines to create the full report
    full_report = "\n".join(report_content)

    # Write to file
    with open(report_file, "w") as f:
        f.write(full_report)

    # Also print to stdout
    print("\n" + full_report + "\n")
    print("=" * 80)


def generate_summary_report(  # noqa: C901
    primary_datasets, reference_datasets, output_dir
):
    """Generate a comprehensive summary report"""
    report_file = f"{output_dir}/comprehensive_stability_summary.txt"

    # Create a string buffer to hold the report content
    report_content = []

    # Header
    report_content.append("Comprehensive Latency Stability Analysis Summary")
    report_content.append("=" * 80)
    report_content.append("")

    # Primary datasets summary
    primary_data = []
    for sheet_name, info in primary_datasets.items():
        model, device_base, os_version = (
            info.get("model", ""),
            info.get("base_device", ""),
            info.get("os_version", ""),
        )
        device_display = (
            f"{device_base}({os_version})" if os_version else info["device"]
        )

        primary_data.append(
            {
                "Dataset": sheet_name,
                "Model": model,
                "Device": device_display,
                "Mean Value": info["metrics"]["mean_raw_latency"],
                "CV (%)": info["metrics"]["cv_raw_latency"],
                "Stability Score": info["metrics"]["stability_score"],
                "Stability Rating": info["metrics"]["stability_rating"],
            }
        )

    primary_df = pd.DataFrame(primary_data).sort_values(
        "Stability Score", ascending=False
    )

    report_content.append("Primary (Private) Datasets Summary:")
    report_content.append(
        tabulate(
            primary_df, headers="keys", tablefmt="grid", floatfmt=".2f", showindex=False
        )
    )
    report_content.append("")

    # Reference datasets summary if available
    if reference_datasets:
        reference_data = []
        for sheet_name, info in reference_datasets.items():
            model, device_base, os_version = (
                info.get("model", ""),
                info.get("base_device", ""),
                info.get("os_version", ""),
            )
            device_display = (
                f"{device_base}({os_version})" if os_version else info["device"]
            )

            reference_data.append(
                {
                    "Dataset": sheet_name,
                    "Model": model,
                    "Device": device_display,
                    "Mean Value": info["metrics"]["mean_raw_latency"],
                    "CV (%)": info["metrics"]["cv_raw_latency"],
                    "Stability Score": info["metrics"]["stability_score"],
                    "Stability Rating": info["metrics"]["stability_rating"],
                }
            )

        reference_df = pd.DataFrame(reference_data).sort_values(
            "Stability Score", ascending=False
        )

        report_content.append("Reference (Public) Datasets Summary:")
        report_content.append(
            tabulate(
                reference_df,
                headers="keys",
                tablefmt="grid",
                floatfmt=".2f",
                showindex=False,
            )
        )
        report_content.append("")

        # Comparison summary for matching datasets
        comparison_data = []
        for _, primary_info in primary_datasets.items():
            for _, ref_info in reference_datasets.items():
                if is_matching_dataset(primary_info, ref_info):
                    primary_metrics = primary_info["metrics"]
                    reference_metrics = ref_info["metrics"]

                    # Extract model and device info for display
                    model, primary_device_base, primary_os = (
                        primary_info.get("model", ""),
                        primary_info.get("base_device", ""),
                        primary_info.get("os_version", ""),
                    )
                    reference_device_base, reference_os = ref_info.get(
                        "base_device", ""
                    ), ref_info.get("os_version", "")

                    primary_device_display = (
                        f"{primary_device_base} ({primary_os})"
                        if primary_os
                        else primary_info["full_device"]
                    )
                    reference_device_display = (
                        f"{reference_device_base} ({reference_os})"
                        if reference_os
                        else ref_info["full_device"]
                    )

                    comparison_data.append(
                        {
                            "Dataset": f"{model} on {primary_device_base}",
                            "Private Device": primary_device_display,
                            "Public Device": reference_device_display,
                            "Private Score": primary_metrics["stability_score"],
                            "Public Score": reference_metrics["stability_score"],
                            "Score Diff": primary_metrics["stability_score"]
                            - reference_metrics["stability_score"],
                            "Private CV (%)": primary_metrics["cv_raw_latency"],
                            "Public CV (%)": reference_metrics["cv_raw_latency"],
                            "CV Diff (%)": primary_metrics["cv_raw_latency"]
                            - reference_metrics["cv_raw_latency"],
                        }
                    )
                    break  # Only use the first matching reference dataset

        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data).sort_values(
                "Score Diff", ascending=False
            )

            report_content.append("Private vs Public Comparison:")
            report_content.append(
                tabulate(
                    comparison_df,
                    headers="keys",
                    tablefmt="grid",
                    floatfmt=".2f",
                    showindex=False,
                )
            )
            report_content.append("")

            # Count datasets where private is better
            private_better_count = sum(
                1 for row in comparison_data if row["Score Diff"] > 0
            )
            public_better_count = sum(
                1 for row in comparison_data if row["Score Diff"] < 0
            )
            equal_count = sum(1 for row in comparison_data if row["Score Diff"] == 0)

            report_content.append(
                f"Private environment is more stable in {private_better_count} of {len(comparison_data)} cases."
            )
            report_content.append(
                f"Public environment is more stable in {public_better_count} of {len(comparison_data)} cases."
            )
            if equal_count > 0:
                report_content.append(
                    f"Both environments show equal stability in {equal_count} of {len(comparison_data)} cases."
                )
            report_content.append("")

    # Overall insights and recommendations
    report_content.append("Overall Insights and Recommendations:")

    # Stability distribution in primary datasets
    stability_counts = primary_df["Stability Rating"].value_counts()
    report_content.append("Stability Distribution in Private Datasets:")
    for rating, count in stability_counts.items():
        report_content.append(f"  - {rating}: {count} dataset(s)")
    report_content.append("")

    # Best configurations
    best_primary = primary_df.iloc[0]
    report_content.append("Best Configurations:")
    report_content.append(
        f"  - Most stable configuration: {best_primary['Dataset']} (Score: {best_primary['Stability Score']:.1f}/100)"
    )
    report_content.append(
        f"    Model: {best_primary['Model']}, Device: {best_primary['Device']}"
    )

    # OS version insights if available
    os_versions = {}
    for sheet_name, info in primary_datasets.items():
        os_version = info.get("os_version", "")
        if os_version:
            os_versions[sheet_name] = os_version

    if os_versions and len(set(os_versions.values())) > 1:
        # Add OS version to primary DataFrame
        primary_df["OS Version"] = primary_df["Dataset"].map(
            lambda x: primary_datasets[x].get("os_version", np.nan)
        )

        # Remove rows with no OS version
        df_os = primary_df[primary_df["OS Version"].notna()]

        if len(df_os) > 0:
            os_stats = (
                df_os.groupby("OS Version")["Stability Score"]
                .mean()
                .sort_values(ascending=False)
            )
            best_os = os_stats.index[0]
            report_content.append(
                f"  - Most stable OS version: {best_os} (Avg. Score: {os_stats.iloc[0]:.1f}/100)"
            )

    # General recommendations
    report_content.append("")
    report_content.append("General Recommendations:")
    report_content.append(
        "  1. For datasets with 'Poor' or 'Moderate' stability, investigate potential causes"
    )
    report_content.append(
        "     such as thermal throttling, background processes, or power management settings."
    )
    report_content.append(
        "  2. Consider increasing warm-up iterations for datasets with high CV values."
    )
    report_content.append(
        "  3. For critical applications, prefer models and devices with 'Good' or 'Excellent' stability."
    )
    if reference_datasets and comparison_data:
        if private_better_count > public_better_count:
            report_content.append(
                "  4. Private environments generally provide better stability and should be preferred"
            )
            report_content.append(
                "     for production deployments where consistent performance is critical."
            )
        elif public_better_count > private_better_count:
            report_content.append(
                "  4. Public environments show better stability in most cases. Consider investigating"
            )
            report_content.append(
                "     factors affecting stability in the private environment."
            )

    # Join all content with newlines to create the full report
    full_report = "\n".join(report_content)

    # Write to file
    with open(report_file, "w") as f:
        f.write(full_report)

    # Also print to stdout
    print("\n" + full_report + "\n")
    print("=" * 80)


def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Analyze ML model latency stability from benchmark data."
    )
    parser.add_argument(
        "--primary-file",
        help="Path to Excel file containing primary (private) benchmark data",
    )
    parser.add_argument(
        "--reference-file",
        help="Path to Excel file containing reference (public) benchmark data for comparison",
        default=None,
    )
    parser.add_argument(
        "--metric",
        help="Target metric to analyze (default: avg_inference_latency(ms)). Examples: avg_inference_latency(ms), token_per_sec",
        default="avg_inference_latency(ms)",
    )
    parser.add_argument(
        "--output-dir",
        default="stability_analysis_results",
        help="Directory to save analysis results (default: stability_analysis_results)",
    )

    parser.add_argument(
        "--verbose-level",
        type=int,
        default=0,
        choices=range(4),
        help="Verbose level 0-3 (default: 0) to control analysis output detail. Higher values show more detailed results.",
    )
    # Parse arguments
    args = parser.parse_args()

    # Run analysis
    analyze_latency_stability(
        args.metric,
        args.primary_file,
        args.reference_file,
        args.output_dir,
        args.verbose_level,
    )


if __name__ == "__main__":
    main()
