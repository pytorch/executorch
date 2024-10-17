# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import re

from typing import List

# Provided by the PyGithub pip package.
from github import Auth, Github
from github.Repository import Repository


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--repo",
        type=str,
        help='The github repo to modify: e.g. "pytorch/executorch".',
        required=True,
    )
    parser.add_argument(
        "--pr",
        type=int,
        help="Number of the PR in the stack to check and create corresponding PR",
        required=True,
    )
    return parser.parse_args()


def extract_stack_from_body(pr_body: str) -> List[int]:
    """Extracts a list of PR numbers from a ghexport-generated PR body.

    The base of the stack is in index 0.
    """

    # Expected format. The `__->__` could appear on any line. Stop parsing
    # after the blank line. This would return [1, 2, 3].
    """
    Stack from [ghstack](https://github.com/ezyang/ghstack) (oldest at bottom):
    * #3
    * __->__ #2
    * #1

    <PR description details>
    """

    prs = []
    ghstack_begin = "Stack from [ghstack](https://github.com/ezyang/ghstack) (oldest at bottom):"
    ghstack_begin_seen = False
    for line in pr_body.splitlines():
        if ghstack_begin in line:
            ghstack_begin_seen = True
        if not ghstack_begin_seen:
            continue
        match = re.match(r"\*(?:.*?)? #(\d+)", line)
        if match:
            # It's a bullet followed by an integer.
            prs.append(int(match.group(1)))
    return list(reversed(prs))


def get_pr_stack_from_number(pr_number: int, repo: Repository) -> List[int]:
    pr_stack = extract_stack_from_body(repo.get_pull(pr_number).body)

    if not pr_stack:
        raise Exception(
            f"Could not find PR stack in body of #{pr_number}. "
            + "Please make sure that the PR was created with ghstack."
        )

    return pr_stack


def create_prs_for_orig_branch(pr_stack: List[int], repo: Repository):
    # For the first PR, we want to merge to `main` branch, and we will update
    # as we go through the stack
    orig_branch_merge_base = "main"
    for i in range(len(pr_stack)):
        pr = repo.get_pull(pr_stack[i])
        if not pr.is_merged():
            print("The PR (and stack above) is not merged yet, skipping")
            # return
        # Check for invariant: For the current PR, it must be gh/user/x/base <- gh/user/x/head
        assert pr.base.ref.replace("base", "head") == pr.head.ref
        # The PR we want to create is then "branch_to_merge" <- gh/user/x/orig
        # gh/user/x/orig is the clean diff between gh/user/x/base <- gh/user/x/head
        orig_branch_merge_head = pr.base.ref.replace("base", "orig")
        bot_metadata = f"""This PR was created by the merge bot to help merge the original PR into the main branch.
ghstack PR number: https://github.com/pytorch/executorch/pull/{pr.number}
^ Please use this as the source of truth for the PR number to reference in comments
ghstack PR base: https://github.com/pytorch/executorch/tree/{pr.base.ref}
ghstack PR head: https://github.com/pytorch/executorch/tree/{pr.head.ref}
Merge bot PR base: https://github.com/pytorch/executorch/tree/{orig_branch_merge_base}
Merge bot PR head: https://github.com/pytorch/executorch/tree/{orig_branch_merge_head}
\nOriginal PR body:\n
        """

        existing_orig_pr = repo.get_pulls(head="pytorch:" + orig_branch_merge_head, base=orig_branch_merge_base, state="open")
        if existing_orig_pr.totalCount > 0:
            print(f"PR for {orig_branch_merge_head} already exists {existing_orig_pr[0]}")
            # We don't need to create/edit because the head PR is merged and orig is finalized.
        else:
            repo.create_pull(
                base=orig_branch_merge_base,
                head=orig_branch_merge_head,
                title=pr.title,
                body=bot_metadata + pr.body,
            )
        # Advance the base for the next PR
        orig_branch_merge_base = orig_branch_merge_head


def main():
    args = parse_args()

    with Github(auth=Auth.Token(os.environ["GITHUB_TOKEN"])) as gh:
        repo = gh.get_repo(args.repo)
        create_prs_for_orig_branch(get_pr_stack_from_number(args.pr, repo), repo)


if __name__ == "__main__":
    main()
