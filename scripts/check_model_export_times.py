#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import re
from collections import defaultdict
from datetime import datetime

import requests


class GithubActionsClient:

    def __init__(self, token: str):

        self.base_url = "https://api.github.com/repos/pytorch/executorch"
        self.__headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github+json",
        }

    def get_runs(self, params=None):

        runs_url = f"{self.base_url}/actions/runs"
        response = requests.get(runs_url, headers=self.__headers, params=params)
        response.raise_for_status()

        return response.json()["workflow_runs"]

    def get_jobs(self, run_id: int, jobs_per_page: int = 100):

        jobs_url = f"{self.base_url}/actions/runs/{run_id}/jobs"
        all_jobs = []
        page = 1

        while True:
            response = requests.get(
                jobs_url,
                headers=self.__headers,
                params={"per_page": jobs_per_page, "page": page},
            )
            response.raise_for_status()

            json_response = response.json()
            jobs = json_response["jobs"]

            if not jobs:  # No more jobs
                break

            all_jobs.extend(jobs)

            # Stop if we got fewer jobs than requested (last page)
            if len(jobs) < jobs_per_page:
                break

            page += 1

        return all_jobs

    def get_job_logs(self, job_id: int):

        logs_url = f"{self.base_url}/actions/jobs/{job_id}/logs"
        response = requests.get(logs_url, headers=self.__headers)
        response.raise_for_status()

        return response.content.decode()


def extract_model_export_times(log):

    duration = re.search(r"Model export completed .* Duration: (\d+)", log)
    docker_image = re.search(r"DOCKER_IMAGE:\s*(.+?)(?:\s|$)", log)
    dtype = re.search(r"DTYPE=(\w+)", log)
    mode = re.search(r"MODE=(\S+)", log)
    runner = re.search(r"runner:\s*(\S+)", log)

    log_extract = {
        "duration": duration.group(1) if duration else None,
        "docker_image": docker_image.group(1) if docker_image else None,
        "dtype": dtype.group(1) if dtype else None,
        "mode": mode.group(1) if mode else None,
        "runner": runner.group(1) if runner else None,
    }

    return log_extract


def extract_full_model_export_times(gha_client, filters=None, run_id=None):

    if run_id:
        # run_id will be a list when using nargs='+'
        if isinstance(run_id, list):
            all_runs = [{"id": rid} for rid in run_id]
        else:
            # Fallback for single string
            all_runs = [{"id": run_id}]
    else:
        # No run_id provided, fetch runs using filters
        all_runs = gha_client.get_runs(params=filters)

    model_tracker = defaultdict(list)

    for idx, run in enumerate(all_runs, 1):

        run_id_val = run["id"]
        print(f"Processing run {idx}/{len(all_runs)}: ID {run_id_val}")

        try:
            jobs = gha_client.get_jobs(run_id_val)

            for job in jobs:

                if job["conclusion"] == "skipped":
                    continue

                if not ("test-llama" in job["name"]):
                    continue

                try:
                    log = gha_client.get_job_logs(job_id=job["id"])

                    extracted_config = extract_model_export_times(log)
                    extracted_config["job_name"] = job["name"]

                    if extracted_config["duration"]:
                        model_tracker[run_id_val].append(extracted_config)

                except Exception as e:
                    print(f"  Warning: Failed to get logs for job {job['id']}: {e}")
                    continue

        except Exception as e:
            print(f"  Error: Failed to get jobs for run {run_id_val}: {e}")
            continue

    return model_tracker


def print_results_as_table(results_dict):
    """Print results as a formatted markdown table."""

    # Extract all jobs from the defaultdict
    all_jobs = []
    for run_id, jobs in results_dict.items():
        for job in jobs:
            job["run_id"] = run_id  # Add run_id to each job
            all_jobs.append(job)

    if not all_jobs:
        print("No jobs found.")
        return

    # Print header
    print("\n## Model Export Times\n")
    print("| Run ID | Job Name | DType | Mode | Runner | Docker Image | Duration (s) |")
    print("|--------|----------|-------|------|--------|--------------|--------------|")

    # Print each job
    for job in all_jobs:
        run_id = job.get("run_id", "N/A")
        job_name = job.get("job_name", "N/A")[:60]  # Truncate long names
        dtype = job.get("dtype", "N/A")
        mode = job.get("mode", "N/A")
        runner = job.get("runner", "N/A")
        docker_image = job.get("docker_image", "None")
        duration = job.get("duration", "N/A")

        # Truncate docker image if too long
        if docker_image and len(docker_image) > 40:
            docker_image = docker_image[:37] + "..."

        print(
            f"| {run_id} | {job_name} | {dtype} | {mode} | {runner} | {docker_image} | {duration} |"
        )

    # Print summary statistics
    print(f"\n**Total Jobs:** {len(all_jobs)}")

    # Calculate average duration
    durations = [
        int(job["duration"]) for job in all_jobs if job.get("duration", "").isdigit()
    ]
    if durations:
        avg_duration = sum(durations) / len(durations)
        print(f"**Average Duration:** {avg_duration:.1f} seconds")
        print(f"**Min Duration:** {min(durations)} seconds")
        print(f"**Max Duration:** {max(durations)} seconds")


def main():

    parser = argparse.ArgumentParser(
        description="A tool to get all model export times for the different configurations based on the githug actions runs"
    )

    parser.add_argument(
        "--github_token",
        metavar="executable",
        type=str,
        help="Your github access token",
        default="",
    )

    parser.add_argument(
        "--created_time",
        metavar="executable",
        type=str,
        help="The date of the earliest github runs to include of the format YYYY-MM-DD",
        default=datetime.today().strftime("%Y-%m-%d"),
    )

    parser.add_argument(
        "--run_id",
        metavar="RUN_ID",
        type=str,
        nargs="+",  # Accept one or more arguments
        help="One or more run IDs to extract model export times from",
        default=None,
    )

    args = parser.parse_args()

    gha_client = GithubActionsClient(token=args.github_token)

    filters = {"created": f">={args.created_time}"}

    model_tracker_output = extract_full_model_export_times(
        gha_client, filters=filters, run_id=args.run_id
    )

    print_results_as_table(model_tracker_output)


if __name__ == "__main__":
    main()
