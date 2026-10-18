import datetime
import os
import sys

from github import Github
from github.GithubException import GithubException

REPO_NAME = "pytorch/executorch"
LABEL = "need-user-input"
REMINDER_MARKER = "<!-- executorch-auto-reminder -->"
CLOSE_MARKER = "<!-- executorch-auto-close -->"

DAYS_BEFORE_REMINDER = 30
DAYS_BEFORE_CLOSE = 30
REMINDER_COOLDOWN_DAYS = 7

DRY_RUN = os.environ.get("DRY_RUN", "true").lower() != "false"
MAX_ACTIONS = int(os.environ.get("MAX_ACTIONS", "25"))
UTC = datetime.timezone.utc

REMINDER_COMMENT = (
    REMINDER_MARKER
    + "\nHi @{user}, this issue/PR has been marked as 'need-user-input'.\n"
    + "Please respond or provide input. If we don't hear back in a month, this will be closed."
)
CLOSE_COMMENT = (
    CLOSE_MARKER
    + "\nClosing due to no response after a month.\n"
    + "If you still need help, feel free to re-open or comment again!"
)


def aware(dt):
    """PyGithub < 2.0 returns naive UTC datetimes."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def timestamp_of(obj):
    """Reviews expose submitted_at, not created_at; pending reviews have neither."""
    ts = getattr(obj, "created_at", None) or getattr(obj, "submitted_at", None)
    return aware(ts) if ts is not None else None


def login_of(obj):
    return getattr(getattr(obj, "user", None), "login", None)


def body_of(obj):
    return getattr(obj, "body", None) or ""


def is_ours(obj):
    body = body_of(obj)
    return REMINDER_MARKER in body or CLOSE_MARKER in body


def label_applied_at(issue):
    """Latest `labeled` event, so removing and re-adding the label restarts the clock."""
    stamps = [
        aware(e.created_at)
        for e in issue.get_events()
        if e.event == "labeled" and e.label and e.label.name == LABEL
    ]
    return max(stamps) if stamps else aware(issue.created_at)


def conversation(issue, pr):
    """Every commentable artifact, as (timestamp, obj) pairs, oldest first."""
    items = list(issue.get_comments())
    if pr is not None:
        items.extend(pr.get_review_comments())
        items.extend(r for r in pr.get_reviews() if body_of(r).strip())

    pairs = []
    for obj in items:
        ts = timestamp_of(obj)
        if ts is not None:
            pairs.append((ts, obj))
    pairs.sort(key=lambda pair: pair[0])
    return pairs


def pushed_since(pr, since, author):
    """True if a commit landed after `since. Uses the *committer* date, not the author date."""
    try:
        commits = pr.get_commits().reversed
    except GithubException:
        return False

    for commit in commits:
        git_commit = getattr(commit, "commit", None)
        committer = getattr(git_commit, "committer", None)
        date = getattr(committer, "date", None)
        if date is None:
            continue
        if aware(date) <= since:
            # .reversed is newest-first, so once we're past the cutoff we're done.
            return False
        pusher = getattr(getattr(commit, "author", None), "login", None)
        # Accept the push when it's the author's, or when GitHub can't attribute
        # it to an account at all (unmatched commit email is common).
        if pusher is None or pusher == author:
            return True
    return False


def process(issue, now):
    number = issue.number
    is_pr = issue.pull_request is not None
    pr = issue.as_pull_request() if is_pr else None
    author = login_of(issue)

    labeled_at = label_applied_at(issue)
    cycle = [(ts, obj) for ts, obj in conversation(issue, pr) if ts > labeled_at]
    reminders = [(ts, obj) for ts, obj in cycle if REMINDER_MARKER in body_of(obj)]

    # --- did the author respond? ---
    replied = author is not None and any(
        login_of(obj) == author and not is_ours(obj) for _, obj in cycle
    )
    pushed = is_pr and not replied and pushed_since(pr, labeled_at, author)

    if replied or pushed:
        how = "replied" if replied else "pushed commits"
        print(f"  #{number}: author {how} -> removing '{LABEL}'", flush=True)
        if DRY_RUN:
            return 0
        issue.remove_from_labels(LABEL)
        return 1

    # --- no reminder posted yet ---
    if not reminders:
        age = (now - labeled_at).days
        if age < DAYS_BEFORE_REMINDER:
            print(
                f"  #{number}: {age}d since label (reminder at {DAYS_BEFORE_REMINDER}d)", 
                flush=True,
            )
            return 0
        if author is None:
            print(
                f"  #{number}: author account unavailable, skipping", 
                flush=True,
            )
            return 0
        print(
            f"  #{number}: first reminder to @{author} ({age}d)", 
            flush=True,
        )
        if DRY_RUN:
            return 0
        issue.create_comment(REMINDER_COMMENT.format(user=author))
        return 1

    # --- reminder exists: close, nudge again, or wait ---
    since_first = (now - reminders[0][0]).days
    if since_first >= DAYS_BEFORE_CLOSE:
        print(
            f"  #{number}: closing ({since_first}d after first reminder)", 
            flush=True,
        )
        if DRY_RUN:
            return 0
        if not any(CLOSE_MARKER in body_of(obj) for _, obj in cycle):
            issue.create_comment(CLOSE_COMMENT)
        issue.edit(state="closed")
        issue.remove_from_labels(LABEL)
        return 1

    if REMINDER_COOLDOWN_DAYS and author:
        since_last = (now - reminders[-1][0]).days
        if since_last >= REMINDER_COOLDOWN_DAYS:
            print(f"  #{number}: follow-up to @{author} (last {since_last}d ago)", flush=True)
            if DRY_RUN:
                return 0
            issue.create_comment(REMINDER_COMMENT.format(user=author))
            return 1

    print(f"  #{number}: waiting ({since_first}d since first reminder)", flush=True)
    return 0

 
def main():
    token = os.environ.get("GH_TOKEN")
    if not token:
        sys.exit("GH_TOKEN is not set.")

    gh = Github(token, per_page=100)
    repo = gh.get_repo(REPO_NAME)
    now = datetime.datetime.now(UTC)

    items = list(repo.get_issues(state="open", labels=[LABEL]))
    n_prs = sum(1 for i in items if i.pull_request is not None)
    print(
        f"[DRY_RUN={DRY_RUN}] {len(items)} open item(s) labeled '{LABEL}' "
        f"({n_prs} PR, {len(items) - n_prs} issue)",
        flush=True,
    )

    actions = failures = 0
    for issue in items:
        if actions >= MAX_ACTIONS:
            print(f"Action cap {MAX_ACTIONS} reached; stopping. Re-run to continue.", flush=True)
            break
        kind = "PR" if issue.pull_request is not None else "issue"
        print(f"Processing {kind} #{issue.number}: {issue.title}", flush=True)
        try:
            actions += process_issue(issue, now)
        except Exception as exc:
            failures += 1
            print(f"  #{issue.number}: failed: {exc!r}", flush=True)

    print(f"Done. actions={actions} failures={failures}", flush=True)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
