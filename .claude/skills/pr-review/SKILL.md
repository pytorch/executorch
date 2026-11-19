---
name: pr-review
description: Review ExecuTorch pull requests for issues CI cannot check. Use when reviewing PRs, when asked to review code changes, or when the user mentions "review PR", "code review", or "check this PR".
---

# ExecuTorch PR Review Skill

Review ExecuTorch pull requests focusing on what CI cannot check, guided by the checklist in [review-checklist.md](review-checklist.md).

## Usage Modes

### No Argument

If the user invokes `/pr-review` with no arguments, **do not perform a review**. Instead, ask the user what they would like to review:

> What would you like me to review?
> - A PR number or URL (e.g., `/pr-review 12345`)
> - A local branch (e.g., `/pr-review branch`)

### Local CLI Mode

The user provides a PR number or URL:

```
/pr-review 12345
/pr-review https://github.com/pytorch/executorch/pull/12345
```

For a detailed review with line-by-line specific comments:

```
/pr-review 12345 detailed
```

Use `gh` CLI to fetch PR data:

```bash
# Get PR details
gh pr view <PR_NUMBER> --json title,body,author,baseRefName,headRefName,files,additions,deletions,commits

# Get the diff
gh pr diff <PR_NUMBER>

# Get PR comments
gh pr view <PR_NUMBER> --json comments,reviews
```

### Local Branch Mode

Review changes in the current branch that are not in `main`:

```
/pr-review branch
/pr-review branch detailed
```

Use git commands to get branch changes:

```bash
# Get current branch name
git branch --show-current

# Get list of changed files compared to main
git diff --name-only main...HEAD

# Get full diff compared to main
git diff main...HEAD

# Get commit log for the branch
git log main..HEAD --oneline

# Get diff stats (files changed, insertions, deletions)
git diff --stat main...HEAD
```

For local branch reviews:
- The "Summary" should describe what the branch changes accomplish based on commit messages and diff
- Use the current branch name in the review header instead of a PR number
- All other review criteria apply the same as PR reviews

### GitHub Actions Mode

When invoked via `@claude /pr-review` on a GitHub PR, the action pre-fetches PR
metadata and injects it into the prompt. Detect this mode by the presence of
`<formatted_context>`, `<pr_or_issue_body>`, and `<comments>` tags in the prompt.

The prompt already contains:
- PR metadata (title, author, branch names, additions/deletions, file count)
- PR body/description
- All comments and review comments (with file/line references)
- List of changed files with paths and change types

Use git commands to get the diff and commit history. The base branch name is in the
prompt context (look for `PR Branch: <head> -> <base>` or the `baseBranch` field).

```bash
# Get the full diff against the base branch
git diff origin/<baseBranch>...HEAD

# Get diff stats
git diff --stat origin/<baseBranch>...HEAD

# Get commit history for this PR
git log origin/<baseBranch>..HEAD --oneline

# If the base branch ref is not available, fetch it first
git fetch origin <baseBranch> --depth=1
```

Do NOT use `gh` CLI commands in this mode -- only git commands are available.
All PR metadata, comments, and reviews are already in the prompt context;
only the diff and commit log need to be fetched via git.

## Review Philosophy

A single line of code can have deep cross-cutting implications. **Treat every line as potentially load-bearing.**

1. **Only report problems** — The review output must contain only issues, concerns, and actionable suggestions. Do NOT mention things that are done correctly, do NOT praise good decisions, do NOT explain why something is fine. If a section has no problems, omit it entirely. The reader's time is precious — every sentence must point to something that needs fixing or further discussion.
2. **Investigate, don't guess** — When uncertain whether a checklist item applies, read the relevant code. A reviewer who guesses wrong provides negative value.
3. **Review the design, not just the implementation** — A PR can have perfectly correct implementation of a bad design. Question side-channel communication, on/off private flags, and demand concrete interface documentation for new contracts between components.
4. **Focus on what CI cannot check** — Don't comment on formatting, linting, type errors, or CI failures. Focus on design quality, interface correctness, thread safety, BC implications, test adequacy, and pattern adherence.
5. **Be specific and actionable** — Reference file paths and line numbers. Name the function/class/file the author should use.
6. **Match the immediate context** — Read how similar features are already implemented in the same file. Pattern mismatches within a file are always wrong.
7. **Assume competence** — The author knows ExecuTorch; explain only non-obvious context.
8. **No repetition** — Each observation appears in exactly one section of the review output.

### Review panel

The review is done by **three independent reviewer sub-agents**, launched in Step 2. They are the only sub-agents in a review. Independent reviewers miss different things, so together they catch more than one pass does, and agreement between them shows which findings are solid. Three is the smallest panel where a majority means something.

- Launch all three in one message, in the foreground (`run_in_background: false`). In non-interactive runs such as GitHub Actions, background sub-agents are stopped when your turn ends and their work is lost.
- Reviewers do not spawn sub-agents of their own. Each one reads the code itself.
- Each reviewer has a budget of about 15 tool calls. It spends them on the changed code, its callers, and the code it uses, then reports.
- Nobody searches the whole git history (for example `git log -S` or `git log -G` over the repository). It is slow, and in GitHub Actions the checkout is shallow, so the history is incomplete anyway.

## Review Workflow

### Step 1: Understand Context

Before reviewing, build understanding of what the PR touches and why:
1. Identify the purpose of the change from title/description/issue
2. Group changes by type (new code, tests, config, docs)
3. Note the scope of changes (files affected, lines changed)

### Step 2: Independent Reviews

Launch the three reviewers (see "Review panel" above) with the same brief. On a PR with many changed files, give each reviewer a different file to start from, so their budgets do not all run out on the same files. Each one does the whole deep review on its own: it goes through **every changed line** in the diff and evaluates it against the review checklist in [review-checklist.md](review-checklist.md), following the Review Philosophy above.

The brief must:
- Carry everything the reviewer needs, because it cannot see your prompt: the PR number or branch, title, description, relevant comments, the base branch, whether a detailed review was asked for, the path `.claude/skills/pr-review/review-checklist.md`, the Review Philosophy rules, the files under "Files to Reference", and the exact commands to get the diff in the current mode (in GitHub Actions mode, only git commands work).
- Include the rules from "Review panel": no sub-agents, the tool call budget, and no whole-history searches.
- Ask the reviewer to find what is wrong with the change, not to confirm that it is fine.
- Contain no findings or opinions of your own, and none from earlier automated reviews of this PR. Human discussion and author replies are fine. Reviewers stay blind to each other.
- Use the `general-purpose` agent type and not set a `model`, so every reviewer runs on the same model and effort as you.
- Ask for a list of findings, not a drafted review. Each finding gives `file:line`, the problem, the concrete trigger (the input, state, or caller that hits it), a severity (blocker, major, or minor), and how it was established (which code was read, or which command was run), marked confirmed or suspected.
- Ask the reviewer to name any changed file it did not get to. For a detailed review, also ask for short notes on naming, wording, and stale comments, for Specific Comments.

Do not review the diff yourself while the reviewers work. Your job is to judge their findings in Steps 3 and 4.

### Step 3: Consolidate Findings

Deduplicate **before** you draft, and before Step 4 as it would slow it down significantly.

Flatten every candidate from all reviewers into a list of `(file:line, one-line claim)` pairs, then collapse it:

- **Same root cause → one finding.** Two reviewers describing one defect from different angles are reporting one defect, not two.
- **Same fix → one finding.** If a single edit resolves several bullets, that is one finding with several consequences.
- **Same `file:line` twice → merge**, unless you can name two independent defects.

Note how many reviewers reported each finding. Two or three is strong evidence the problem is real. A finding from a single reviewer is not wrong for that reason, but it needs a closer look in Step 4.

If a changed file was reached by no reviewer, review that file yourself against the checklist before Step 4, with a few reads per file. List any file you still could not get to in the Summary, so the review does not look more complete than it is.

Only then assign each survivor to exactly one section (see "One finding, one section" under Output Format) and write it up. Every finding must be traceable to a specific line in the diff.

### Step 4: Fact-Check

Fact-check the **consolidated** list yourself, without sub-agents. For every finding you would post, re-read the cited code, usually one or two reads per finding, and mark it **valid**, **invalid**, or **needs rewording**. Check findings that all reviewers agree on too, because agreement is not proof. For a finding from a single reviewer, also repeat the read or command it cites. Do not start new investigations or look for new problems here. Drop invalid issues, reword the rest. If unsure, leave the issue with a comment for the author that this is low confidence.

## Output Format

Structure your review as follows. **Omit sections where you have no problems to report** — most reviews should only have a few sections. Do not write "No concerns", "Looks good", or any affirmative commentary. Every sentence in the review must identify a problem or request a change.

The Summary section is the one exception: it should briefly state what the PR does (1 sentence) and then state the problems found, or explicitly say no issues were found.

```markdown
## PR Review: #<number>
<!-- Or for local branch reviews: -->
## Branch Review: <branch-name> (vs main)

### Summary
What the PR does (1 sentence), then the overall verdict.

### Code Quality
[Problems only]

### Infrastructure
[Problems only — flag checklist items that are violated]

### Testing
[Problems only — missing tests, wrong patterns, inadequate coverage]

### API Design
[Problems only]

### Security
[Problems only]

### Thread Safety
[Problems only]

### Backward Compatibility
[Problems only]

### Performance
[Problems only]

### Recommendation
**Approve** / **Request Changes** / **Needs Discussion**

Confirmed unnecessary use of GitHub Actions storage (see the confirmation bar in [review-checklist.md](review-checklist.md)) always means **Request Changes**. Unconfirmed storage questions do not.

[Brief justification — focus on what blocks approval, if anything]
```

**One finding, one section.** The categories overlap by construction — a missing test for a bad public API is legitimately API Design, Testing, and Code Quality. Assign each finding to exactly one section, first match wins:

1. **Security** — has a security consequence
2. **Thread Safety** — has a concurrency consequence
3. **Backward Compatibility** — breaks or risks breaking existing callers
4. **API Design** — the defect is in a public or frozen interface (names, signatures, documented semantics)
5. **Infrastructure** — a missing or incorrect hook into an existing ExecuTorch or CI subsystem
6. **Testing** — the defect is *only* absent or inadequate coverage
7. **Performance**
8. **Code Quality** — everything else

State the finding's full consequence once, in its assigned section. When it genuinely spans categories, say so inline in that one bullet ("this is also a BC break for out-of-tree backends") rather than adding a second bullet under the other section. GitHub Actions storage findings are **Infrastructure** findings. Confirmed waste must receive the same severity and visibility as a functional bug.

This precedence governs the eight finding sections only. Summary and Recommendation are not finding buckets — see their rules in the template above. Neither is Specific Comments — see its rules below.

Do not invent sections outside this template. File a finding by its *consequence*, not its file type: a docstring that misstates operator semantics is API Design, not documentation. Where [review-checklist.md](review-checklist.md) groups or names a topic differently, that organizes the checklist, not the output; this precedence wins.

### Specific Comments (Detailed Review Only)

**Only include this section if the user requests a "detailed" or "in depth" review.**

**Do not repeat observations already made in other sections.** This section is for points too localized to carry their own categorized finding — single-line naming, wording, or stale-comment notes. Anything with a behavioral or interface consequence belongs in one of the eight sections above.

When requested, add file-specific feedback with line references:

```markdown
### Specific Comments
- `exir/example/module.py:78` - reuses the name of the enclosing loop variable
- `runtime/example/example.cpp:203` - the error message says "tensor", but the parameter is a list
- `backends/example/partitioner.py:512` - stale comment: describes the pre-refactor control flow
```

## Files to Reference

When reviewing, consult these project files for context — read them rather than relying on memory, as they change frequently:
- `CLAUDE.md` - Available skills and repository conventions
- `CONTRIBUTING.md` - PR requirements and review process
