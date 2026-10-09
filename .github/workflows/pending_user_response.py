import datetime
import os

from github import Github

REPO_NAME = "pytorch/executorch"
LABEL = "need-user-input"
REMINDER_MARKER = "<!-- executorch-auto-reminder -->"

DAYS_BEFORE_REMINDER = 30
DAYS_BEFORE_CLOSE = 30
REMINDER_COOLDOWN_DAYS = 7
DRY_RUN = os.environ.get("DRY_RUN", "true").lower() == "true"

REMINDER_COMMENT = (
    REMINDER_MARKER
    + f"This issue/PR has been marked as 'need-user-input'. "
    + f"Please respond or provide input. If we don't hear back in {DAYS_BEFORE_REMINDER} days, this will be closed."
)
CLOSE_COMMENT = (
    REMINDER_MARKER
    + f"\nClosing due to no response after {DAYS_BEFORE_CLOSE} days. "
    + "If you still need help, feel free to re-open or comment again!"
)


def main():
    g = Github(os.environ["GH_TOKEN"])
    repo = g.get_repo(REPO_NAME)

    print(f"[DRY_RUN={DRY_RUN}] Fetching open issues with label '{LABEL}'.")
    issues = repo.get_issues(state="open", labels=[LABEL])

    now = datetime.datetime.now(datetime.timezone.utc)

    for issue in issues:
        print(f"Processing issue/PR #{issue.number}: {issue.title}")

        comments = sorted(
            issue.get_comments(),
            key=lambda comment: comment.created_at,
        )

        labeled_at = max(
            (
                e.created_at
                for e in issue.get_events()
                if e.event == "labeled"
                and e.label
                and e.label.name == LABEL
            ),
            default=issue.created_at,
        )
        cycle = [c for c in comments if c.created_at > labeled_at]
        reminders = [c for c in cycle if REMINDER_MARKER in (c.body or "")]

        # ---- AUTOMATION LOGIC ----
        if any(
            c.user.login == issue.user.login
            for c in cycle
            if REMINDER_MARKER not in (c.body or "")
        ):
            print(f"User responded; remove '{LABEL}' label")
            if not DRY_RUN:
                issue.remove_from_labels(LABEL)
            continue
        elif (now - reminders[0].created_at).days >= DAYS_BEFORE_CLOSE:
            print(f"Close issue/PR due to inactivity")
            if not DRY_RUN:
                issue.create_comment(CLOSE_COMMENT)
                issue.edit(state="closed")
                issue.remove_from_labels(LABEL)
        elif not reminders:
            if (now - labeled_at).days >= DAYS_BEFORE_REMINDER:
                user = issue.user.login
                print(f"Posting first reminder for {user}")
                message = REMINDER_COMMENT.format(user)
                if not DRY_RUN:
                    issue.create_comment(message)
        elif (now - reminders[-1].created_at).days >= REMINDER_COOLDOWN_DAYS:
            user = issue.user.login
            message = REMINDER_COMMENT.format(user)
            print(f"Posting follow-up reminder for {user}")
            if not DRY_RUN:
                issue.create_comment(message)
        


if __name__ == "__main__":
    main()
