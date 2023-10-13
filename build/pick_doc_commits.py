#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Helps find commits to cherrypick into a release branch.

Usage:
  pick_doc_commits.py --main=origin/main --release=origin/release/5.5

It will find commits on the main branch that are not on the release branch, and
filter them down to the docs-only commits that should be cherrypicked. It will
also print the commits that were filtered out.

This tool will not actually modify the git repo, it will only print the commands
to run.

Must be run from inside the repo, ideally after a recent `git pull`. Does not
care which branch is currently checked out.
"""

import argparse
import datetime
import re
import subprocess
import sys
import textwrap
from typing import List


# The script will print extra info when this is > 0, and more at higher levels.
# Controlled by the --verbose flag.
verbosity = 0


def debug_log(message: str):
    """Prints a message to stderr if verbosity is greater than zero."""
    global verbosity
    if verbosity > 0:
        sys.stderr.write(f"VERBOSE: {message}\n")


def run_git(command: List[str]) -> List[str]:
    """Runs a git command and returns its stdout as a list of lines.

    Prints the command and its output to debug_log() if verbosity is greater
    than 1.

    Args:
        command: The args to pass to `git`, without the leading `git` itself.
    Returns:
        A list of the non-empty lines printed to stdout, without trailing
        newlines.
    Raises:
        Exception: The command failed.
    """
    try:
        if verbosity > 1:  # Higher verbosity required
            debug_log("Running command: 'git " + " ".join(command) + "'")
        result = subprocess.run(["git", *command], capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Error running command '{command}':\n{result.stderr}")
        lines = result.stdout.split("\n")
        # Remove empty and whitespace-only lines.
        lines = [line.strip() for line in lines if line.strip()]
        global verbose
        if verbosity > 1:
            debug_log("-----BEGIN GIT OUTPUT-----")
            for line in lines:
                debug_log(line)
            debug_log("-----END GIT OUTPUT-----")
        return lines
    except Exception as e:
        raise Exception(f"Error running command '{command}': {e}")


class Commit:
    """A git commit hash and its one-line message."""

    def __init__(self, hash: str, message: str = ""):
        """Creates a new Commit with the given hash.

        Args:
            hash: The hexadecimal hash of the commit.
            message: The one-line summary of the commit. If empty, this method
                will ask git for the commit message.
        """
        self.hash = hash.strip()
        if not message:
            # Ask git for the commit message.
            lines = run_git(["log", "-1", "--pretty=%s", self.hash])
            # Should just be one line, but could be zero.
            message = " ".join(lines)
        self.message = message.strip()

    @staticmethod
    def from_line(line: str) -> "Commit":
        """Creates a Commit from a string of the form '<hash> [<message>]'."""
        parts = line.split(" ", maxsplit=1)
        parts = [part.strip() for part in parts if part.strip()]
        assert len(parts) >= 1, f"Expected at least one part in line '{line}'"
        return Commit(hash=parts[0], message=parts[1] if len(parts) > 1 else "")

    def __repr__(self):
        return f"Commit('{self.hash[:8]}', '{self.message}')"

    def __str__(self):
        return f"{self.hash[:8]} {self.message}"


def is_doc_only_commit(commit: Commit) -> bool:
    """Returns True if the commit only touched "documentation files"."""

    def is_doc_file(path: str) -> bool:
        """Returns true if the path is considered to be a "documentation file"."""
        return (
            # Everything under docs, regardless of the file type.
            path.startswith("docs/")
            # Any markdown or RST file in the repo.
            or path.endswith(".md")
            or path.endswith(".rst")
        )

    # The first line is the full hash, and the rest are the files modified by
    # the commit, relative to the root of the repo.
    lines = run_git(["diff-tree", "--name-only", "-r", commit.hash])
    all_files = frozenset(lines[1:])
    doc_files = frozenset(filter(is_doc_file, all_files))
    non_doc_files = all_files - doc_files
    is_doc_only = all_files == doc_files

    if verbosity > 0 and not is_doc_only:
        debug_log(
            f"{repr(commit)} touches {len(non_doc_files)} non-doc files, "
            + f"like '{sorted(non_doc_files)[0]}'."
        )

    return is_doc_only


def print_wrapped(text: str, width: int = 80) -> None:
    """Print text wrapped to fit within the given width.

    Indents additional lines by four spaces.
    """
    print("\n    ".join(textwrap.wrap(text, width=width - 4, break_on_hyphens=False)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prints differences between git branches."
    )
    parser.add_argument(
        "--main",
        default="origin/main",
        type=str,
        help="The name of the main (source) branch to pick commits from.",
    )
    parser.add_argument(
        "--release",
        type=str,
        help="The name of the release (destination) branch to pick commits onto, "
        + "ideally with the 'origin/' prefix",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Log extra output. Specify more times (-vv) for more output.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    main_branch = args.main
    release_branch = args.release

    global verbosity
    verbosity = args.verbose

    # Returns a list of hashes that are on the main branch but not the release
    # branch. Each hash is preceded by `+ ` if the commit has not been cherry
    # picked onto the release branch, or `- ` if it has.
    cherry_lines = run_git(["cherry", release_branch, main_branch])
    print_wrapped(
        f"Commits on '{main_branch}' that have already been cherry-picked into '{release_branch}':"
    )
    if not cherry_lines:
        print("- <none>")
    candidate_commits = []
    for line in cherry_lines:
        commit = Commit.from_line(line[2:])
        if line.startswith("+ "):
            candidate_commits.append(commit)
        elif line.startswith("- "):
            print(f"- {commit}")
    print("")

    # Filter out and print the commits that touch non-documentation files.
    print_wrapped(
        f"Will not pick these commits on '{main_branch}' that touch non-documentation files:"
    )
    if not candidate_commits:
        print("- <none>")
    doc_only_commits = []
    for commit in candidate_commits:
        if is_doc_only_commit(commit):
            doc_only_commits.append(commit)
        else:
            print(f"- {commit}")
    print("")

    # Print the commits to cherry-pick.
    print_wrapped(
        f"Remaining '{main_branch}' commits that touch only documentation files; "
        + f"will be cherry-picked into '{release_branch}':"
    )
    if not doc_only_commits:
        print("- <none>")
    for commit in doc_only_commits:
        print(f"- {commit}")
    print("")

    # Print instructions for cherry-picking the commits.
    if doc_only_commits:
        # Recommend a unique branch name.
        suffix = datetime.datetime.utcnow().strftime("%Y%m%d%H%M")
        branch_name = "cherrypick-" + release_branch.replace("/", "-") + "-" + suffix

        print("Cherry pick by running the commands:")
        print("```")
        print(f"git checkout {release_branch}")
        print(
            # Split lines with backslashes to make long lists more legible but
            # still copy-pasteable.
            "git cherry-pick \\\n  "
            + " \\\n  ".join([commit.hash for commit in doc_only_commits])
        )
        print(f"git checkout -b {branch_name}")
        print("```")
        print("")
        print("To verify that this worked, re-run this script with the arguments:")
        print("```")
        print(f"--main={main_branch} --release={branch_name}")
        print("```")
        print("It should show no doc-only commits to cherry-pick.")
        print("")
        print(f"Then, push {branch_name} to GitHub:")
        print("```")
        print(f"git push --set-upstream origin {branch_name}")
        print("```")
        print("")
        print_wrapped(
            "When creating the PR, remember to set the 'into' branch to be "
            # Remove "origin/" if present since it won't appear in the GitHub
            # UI.
            + f"'{re.sub('^origin/', '', release_branch)}'."
        )
    else:
        print_wrapped(
            "It looks like there are no doc-only commits "
            + f"on '{main_branch}' to cherry-pick into '{release_branch}'."
        )


if __name__ == "__main__":
    main()
